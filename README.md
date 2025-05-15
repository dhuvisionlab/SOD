# YOLO-SOD<br>
This repos contains the official codes for papers:<br>
<br>
**A deep neural network for small object detection in complex environments 
with unmanned aerial vehicle imagery**<br>
Sayed Jobaer, Xue-song Tang, Yihong Zhang <be>
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
***<p align="center">The detailed demonstration of several key modules in YOLO-SOD***<br><br>
<img src="images/2.png" width="90%" height="90%"><br><br>
***<p align="center">YOLO-SOD detection on images that have multiple objects of the same image***<br><br>
<img src="images/3.png" width="90%" height="90%"><br><br>

## Quick Start<br>

```bash
git clone https://github.com/dhuvisionlab/SOD.git # clone 
cd YOLO-SOD
pip install -r requirements.txt  # install
```


## Original_Dataset<br>
The SOD dataset’s data was carefully gathered in 2022 at Songjiang University Town, Shanghai, China. The images were captured under varying weather conditions and at different times to ensure diversity and real-world relevance. The achievement was conducted using a high- performance ‘DJI MAVIC AIR 2’ drone, at an altitude of 100–200 m, resulting in 3000 images encompassing objects of varying sizes.<br><br>
***<p align="center">Some sample images from our SOD-Dataset***<br><br>
<img src="images/4.png" width="90%" height="90%"><br><br>

### Data_preprocessing<br>
Raw data needs to be preprocessed before it can be fed into networks for training or testing. First, we apply image pre-processing methods such as brightness correction and image filtering on sample images to enhance the quality of the dataset. Then, an annotation software called ‘LabelImg’ was used to draw the ground truth bounding boxes of the disease or pests in all images. Visit this link to download the LabelImg: https://github.com/HumanSignal/labelImg <be>

### Data_Download<br>
Visit this link to download the dataset: https://drive.google.com/drive/folders/1OuoB48SMy5MPwzQ3Fm6cfGvgRINe07-T?usp=drive_link

## Acknowledgement
Part of our code was descended and modified from the open-source code by ultralytics. Their original code can be found at: [https://github.com/ultralytics/yolov5.git].


## Citation<br>
Please consider citing our papers if the project helps your research with the following BibTex:
```
@article{JOBAER2025110466,
title = {A deep neural network for small object detection in complex environments with unmanned aerial vehicle imagery},
journal = {Engineering Applications of Artificial Intelligence},
volume = {148},
pages = {110466},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.110466},
url = {https://www.sciencedirect.com/science/article/pii/S095219762500466X},
author = {Sayed Jobaer and Xue-song Tang and Yihong Zhang},
}
```
