

from ctypes import windll
import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
import torch.backends.cudnn as cudnn
import time

import math

sys.path.insert(0, './yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from threading import Lock, Thread
from time import sleep


import cv_viewer.tracking_viewer as cv_viewer
import cv_viewer.mesh_viewer as mesh_v


class Detector:

    lock = Lock()
    image_net = None
    run_signal = False
    exit_signal = False
    weights='weights\\best.pt'
    svo=None
    img_size=416
    conf_thres=0.4
    detections = None
    IMG = None
    TRACK_IMG = None
    window = None
    IMGWithFloor = None
    mesh_viewer=None
    camera_infos=None
    ctrl = False


    def __init__(self, window, weights='weights\\best.pt', svo=None, img_size=416, conf_thres=0.4):
        self.window=window
        self.weights=weights
        self.svo=svo
        self.conf_thres=conf_thres
        self.img_size=img_size
        self.main()


    def img_preprocess(self, img, device, half, net_size):
        net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
        net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        net_image = np.ascontiguousarray(net_image)

        img = torch.from_numpy(net_image).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, ratio, pad


    def xywh2abcd(self, xywh, im_shape):
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
        x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
        y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
        y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_min
        output[2][1] = y_max

        output[3][0] = x_max
        output[3][1] = y_max
        return output


    def detections_to_custom_box(self, detections, im, im0):
        output = []
        for i, det in enumerate(detections):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

  
                    obj = sl.CustomBoxObjectData()
                    obj.bounding_box_2d = self.xywh2abcd(xywh, im0.shape)
                    obj.label = cls
                    obj.probability = conf
                    obj.is_grounded = False
                    output.append(obj)
        return output


    def meshViewerInit(self):
        has_imu =  self.camera_infos.sensors_configuration.gyroscope_parameters.is_available
        self.mesh_viewer = mesh_v.GLViewer()
        self.mesh_viewer.setDetector(self)
        self.mesh_viewer.init(self.camera_infos.camera_configuration.calibration_parameters.left_cam, has_imu)


    def torch_thread(self, weights, img_size, conf_thres=0.2, iou_thres=0.45):

        print("Intializing Network...")

        device = select_device()

        half = device.type != 'cpu'  # half precision only supported on CUDA
        imgsz = img_size

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32
        stride = int(model.stride.max())  
        imgsz = check_img_size(imgsz, s=stride)  
        if half:
            model.half()  # to FP16
        cudnn.benchmark = True


        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  

        while not self.exit_signal:
            if self.run_signal:
                self.lock.acquire()
                img, ratio, pad = self.img_preprocess(self.image_net, device, half, imgsz)

                pred = model(img)[0]
                det = non_max_suppression(pred, conf_thres, iou_thres)

                self.detections = self.detections_to_custom_box(det, img, self.image_net)
                self.lock.release()
                self.run_signal = False
            sleep(0.01)


    def main(self):

        capture_thread = Thread(target=self.torch_thread,
                                kwargs={'weights': self.weights, 'img_size': self.img_size, "conf_thres": self.conf_thres})
        capture_thread.start()

        print("Initializing Camera...")

        zed = sl.Camera()

        input_type = sl.InputType()
        if self.svo is not None:
            input_type.set_from_svo_file(self.svo)


        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 20

        runtime_params = sl.RuntimeParameters()

        status = zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        image_left_tmp = sl.Mat()

        print("Initialized Camera")

        positional_tracking_parameters = sl.PositionalTrackingParameters()

        zed.enable_positional_tracking(positional_tracking_parameters)

        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        zed.enable_object_detection(obj_param)

        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()


        self.camera_infos = zed.get_camera_information()

        point_cloud_res = sl.Resolution(min(self.camera_infos.camera_resolution.width, 720),
                                        min(self.camera_infos.camera_resolution.height, 404))
        point_cloud_render = sl.Mat()
        
        point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        image_left = sl.Mat()

        display_resolution = sl.Resolution(min(self.camera_infos.camera_resolution.width, 1280),
                                        min(self.camera_infos.camera_resolution.height, 720))
        image_scale = [display_resolution.width / self.camera_infos.camera_resolution.width, display_resolution.height / self.camera_infos.camera_resolution.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)


        camera_config = zed.get_camera_information().camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps,
                                                        init_params.depth_maximum_distance)

        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

        pose = sl.Pose()


        while not self.exit_signal:   
            DistanceToDamege=[]
           
            xt=True

            if xt:
                if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

                    self.lock.acquire()
                    zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                    self.image_net = image_left_tmp.get_data()
                    self.lock.release()
                    self.run_signal = True

                    while self.run_signal:
                        sleep(0.001)

                    
                    self.lock.acquire()

                    zed.ingest_custom_box_objects(self.detections)
                    self.lock.release()
                    zed.retrieve_objects(objects, obj_runtime_param)


                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
                    point_cloud.copy_to(point_cloud_render)
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                    zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

                  
                    np.copyto(image_left_ocv, image_left.get_data())
                    try:
                      
                        for l in objects.object_list:
                            DistanceToDamege.append([l.dimensions[1], l.position[2]])      # l.position[1] -> y ekseninde olan uzaklık, l.position[2] -> Aracın çukura olan uzaklığı
                    except:                                                                # l.dimensions[1] -> Cismin boyu(dikeyde)
                        pass
                    self.window.inputs.clear.emit()
                    if DistanceToDamege is not None:
                        i=1
                        for distance in DistanceToDamege:
                            if not math.isnan(distance[0]) and not math.isnan(distance[1]):
                                distToRoad = round(((distance[0])*(100)),2)
                                distToRoad_txt = str(distToRoad)+ " cm"
    
                                distToCar = (round(distance[1],3))*-1
                                distToCar_txt = str(distToCar)+" m"

                                if distToCar > 5:
                                    rankOfDanger = 0
                                elif distToCar < 5:
                                    if (distToRoad>=0 and distToRoad < 7.5) or (distToRoad<=0 and distToRoad > -7.5):
                                        rankOfDanger = 1
                                    else:
                                        rankOfDanger = 2

                                self.window.inputs.input.emit([i, distToRoad_txt, distToCar_txt, rankOfDanger])
                                i+=1

                    cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)

                    # Tracking view
                    track_view_generator.generate_view(objects, pose, image_track_ocv, objects.is_tracked)


                    self.TRACK_IMG = cv2.resize(image_track_ocv, (0,0), fx=0.5, fy=0.5) 
                    image_left_ocv=cv2.circle(image_left_ocv, (626,358), 2,255,4)
                    self.IMG = cv2.resize(image_left_ocv, (0,0), fx=0.5, fy=0.5) 
                    self.window.displayImage(self.window.trackImage, self.TRACK_IMG, 1)
                    self.window.displayImage(self.window.oriImage, self.IMG, 1)

                else:
                    self.exit_signal = True


        self.mesh_viewer.exit()
        self.exit_signal = True
        zed.disable_positional_tracking()
        zed.close()