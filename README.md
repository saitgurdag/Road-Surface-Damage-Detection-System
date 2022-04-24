# Road-Surface-Damage-Detection
## Setting up

 - Clone Yolov5 into the current folder

```sh
git clone https://github.com/ultralytics/yolov5
# Install the dependencies if needed
cd yolov5
pip install -r requirements.txt
```

--prepare your own model with road surface damage dataset:

The datasets were prepared by examining the following studies:

RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification[1]
Road surface detection and differentiation considering surface damages [2]
Detecting potholes using simple image processing techniques and real-world footage [3]
A Comparison of Low-Cost Monocular Vision Techniques for Pothole Distance Estimation [4]

 https://app.roboflow.com/ds/WcKTZGNMLH?key=5P7rZneRe4
 https://app.roboflow.com/ds/wb8wBYlx1A?key=fN5eRMtTo5
 https://app.roboflow.com/ds/1qwpYECJnh?key=neIer5kjnt
 https://app.roboflow.com/ds/4aKVya7MsB?key=nxH4XZoUqn
 https://app.roboflow.com/ds/J3Bw4oBvcn?key=gmxO34NukI
 https://app.roboflow.com/ds/Xe5vXh2hbt?key=EE8IvalZMO
 https://app.roboflow.com/ds/D2oir8xE2f?key=kjb0X7PmRb

 --references

[1] Arya, Deeksha; Maeda, Hiroya; Ghosh, Sanjay Kumar; Toshniwal, Durga ; Omata, Hiroshi ; Kashiyama, Takehiro ;  Seto, Toshikazu; Mraz, Alexander ; Sekimoto, Yoshihide (2021), “RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification”, Mendeley Data, V1, doi: 10.17632/5ty2wb6gvg.1
[2]Rateke,Thiago;Von Wangenheim,Aldo;”Road surface detection and differentiation considering surface damages”,doi:10.1007/s10514-020-09964-3
[3] S. Nienaber, M.J. Booysen, R.S. Kroon, “Detecting potholes using simple image processing techniques and real-world footage”, SATC, July 2015, Pretoria, South Africa
[4] S. Nienaber, R.S. Kroon, M.J. Booysen , “A Comparison of Low-Cost Monocular Vision Techniques for Pothole Distance Estimation”, IEEE CIVTS, December 2015, Cape Town, South Africa.

```