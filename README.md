# YOLO-SOD<br>
This repos contains the official codes for papers:<br>
<br>
**A deep neural network for small object detection in complex environments 
with unmanned aerial vehicle imagery**<br>
Sayed Jobaer, Xue-song Tang, Yihong Zhang<br
Pubilshed on *Engineering Applications of Artificial Intelligence* in 2025<br>
[[Paper](https://doi.org/10.1016/j.engappai.2025.110466)]<br>
<br>

## Prerequisites<br>
The code written in pytorch, and their corresponding configurations are as follows:
* All deep networks run under Ubuntu 20.04
  * Python >=3.9
  * Pytorch >=1.8.0 

## Introduction<br>

Deep learning-based object detectors perform effectively on edge devices but encounter challenges with small and flat objects in complex environments, especially under low-light conditions and in high-altitude images
captured by unmanned aerial vehicles (UAVs). The primary issue is the pixel similarity between objects and their backgrounds, making detection challenging. While existing detectors struggle to detect small and flat objects in these scenarios, the advent of you only look once (YOLO) algorithms have shown promise. However, they still have limitations in detecting small and flat objects under these conditions. Due to a shortage of suitable datasets covering complex environments and lighting conditions, the field lacks comprehensive research on detecting small and flat objects in UAV-assisted images.<br><be>

To address these issues, we develop a dataset with nine classes tailored to small object detection (SOD) challenges. We propose a dynamic model based on the you only look once network v5 (version 6.2) architecture to overcome the above-mentioned limitations.<br>
***<p align="center">Architecture of YOLO-SOD***<br><br>
<img src="images/1.png" width="90%" height="100%"><br><br>
***<p align="center">The detailed demonstration of several key modules in YOLO-JD***<br><br>
<img src="images/2.png" width="90%" height="90%"><br><br>
***<p align="center">YOLO-JD detection on images that have multiple instances of the same disease and that have multiple classes of diseases and pests on the same image***<br><br>
<img src="images/3.png" width="90%" height="90%"><br><br>

## Quick Start<br>

```bash
git clone https://github.com/foysalahmed10/YOLO-JD.git # clone
cd YOLO-JD-master
pip install -r requirements.txt  # install
```


## Original_Dataset<br>
The images of jute diseases and pests were collected in Jamalpur and Narail districts in Bangladesh in July 2021. To diversify the dataset, the images were captured over the course of a single day under both sunny and cloudy weather. The images were captured by a Canon Powershot G16 camera and the camera of a Samsung Galaxy S10 with different viewing angles and different distances (0.3–0.5 m). In total, 4418 images in multiple jute disease and pest classes were obtained. The light intensity and background circumstances of the images vary greatly in the dataset. Though the image sizes are not uniform in our dataset, we prepare a normalization step at the beginning of the network to unify all images to a fixed resolution of 640 × 640. Eight common diseases including stem rot, anthracnose, black band, soft rot, tip blight, dieback, jute mosaic, and jute chlorosis, as well as two pests—Jute Hairy Caterpillar, and Comophila sabulifers—are incorporated into our dataset.<br><br>
***<p align="center">Some sample images from our jute diseases and pests dataset***<br><br>
<img src="images/4.png" width="90%" height="90%"><br><br>

### Data_preprocessing<br>
Raw data needs to be preprocessed before it can be fed into networks for training or testing. First, we apply image pre-processing methods such as brightness correction and image filtering on sample images to enhance the quality of the dataset. Then, an annotation software called ‘LabelImg’ was used to draw the ground truth bounding boxes of the disease or pests in all images. Visit this link to download the LabelImg: https://github.com/HumanSignal/labelImg <be>

### Data_Download<br>
Visit this link to download the dataset: https://1drv.ms/u/s!Al1NYDOSIj467ysFMpzJWJc7fEtc?e=FIU6bX

## Acknowledgement
Part of our code was descended and modified from the open-source code by ultralytics. Their original code can be found at: [https://github.com/ultralytics/yolov5.git].


## Citation<br>
Please consider citing our papers if the project helps your research with the following BibTex:
```
@article{li2022plantnet,
  title={YOLO-JD: A Deep Learning Network for Jute Diseases and Pests Detection from Images},
  author={Li, Dawei and Ahmed, Foysal and Wu, Nailong and Sethi, A.Ishrat},
  journal={Plants},
  volume={11(7)},
  pages={937},
  year={2022},
  publisher={MDPI}
  issn = {2223-7747},
  doi = {https://doi.org/10.3390/plants11070937}
}
```
