# YOLO
<H2>使用yolo模型</h2>
  

<h4>安裝OpenCV</h4>

```
brew install opencv@3
brew link –force opencv@3
brew install libomp
brew install clblas
brew install clfft
brew install cmake
brew install wget
```

使用 CMD 從 YOLOv4的發表者 AlexeyAB  https://github.com/AlexeyAB/darknet 下載 YOLOv4的資料
```
cd <欲將darknet放置的路徑>
git clone https://github.com/AlexeyAB/darknet.git
``` 

修改 darknet 資料夾底下  makefile

```
GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=0
AVX=0
OPENMP=0
LIBSO=0
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
因為我的 macbook pro 沒有 GPU 因此只將 OPENCV=0 修改為 OPENCV=1，如此就能夠用 CPU 與 OpenCV 執行 darknet
```
GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=0
LIBSO=0
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```

使用 CMD 建立 darknet 的 UNIX 執行檔

```
make
```


原始的YOLO模型是使用COCO資料集進行建模
```
                                    <載入coco.data>             <載入YOLOv4模型>        <載入模型權重>          <欲識別圖片>
./darknet detector train ./build/darknet/x64/data/coco.data ./cfg/yolov4-tiny.cfg ./yolov4-tiny.conv.29   ./data/dog.jpg
```

<H2>訓練yolo模型</h2>

<h4>影像標記</h4>

```
pip install labelImg

# 執行 LabelImg
labelImg
```

<h4>前置作業</H4>

在Darknet 資料夾底下建立一個名叫 retrain 資料夾，該資料夾底下分別在建立 image ,cfg ,weights 資料夾

* cfg 資料夾設置
  * obj.data : 記載darknet 的 UNIX 執行檔所需資料路徑 \
  <img src=https://github.com/DaYi-TW/YOLO/blob/main/obj.png alt="drawing" height="300" ></img>
  * obj.name : 欲分類類別
  * yolov4-tiny.cfg : YOLOv4模型
  * train.txt : 紀錄訓練影像路徑
  * test.txt : 紀錄測試影像路徑
* image 資料夾設置
  * jpg : 放欲訓練及測試影像
  * txt : 影像標記後的座標
* weights 資料夾設置
  * 存放訓練前模型權重
  * 存放訓練後模型權重

<h4>本地訓練</H4>

```
cd <darknet的位置下>
```
Training the model
```
                               <載入obj.data>             <載入YOLOv4模型>           <載入預先訓練好的模型權重>          
./darknet detector train ./retrain/cfg/obj.data ./retrain/cfg/yolov4-tiny.cfg ./retrain/yolov4-tiny.conv.29   -dont_show -clear
```
<h4>Google Colab訓練</H4>



```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
!ln -s '/content/gdrive/My Drive/spaceColab' /
%cd /spaceColab/darknet
!ls -la spaceColab/darknet
!chmod 777 darknet
```

Training the model
```
                               <載入obj.data>             <載入YOLOv4模型>           <載入預先訓練好的模型權重>          
!./darknet detector train ./retrain/cfg/obj.data ./retrain/cfg/yolov4-tiny.cfg ./retrain/yolov4-tiny.conv.29   -dont_show -clear
```
