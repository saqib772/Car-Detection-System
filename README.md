# Vehicle Detection using Cascade Classifier and YOLO

This repository contains Python scripts for vehicle detection in videos using two different methods: Cascade Classifier and YOLO (You Only Look Once) classifier.

# Files:
main.py: Vehicle detection using Cascade Classifier. It utilizes the OpenCV Haar Cascade Classifier for car detection in videos.

yolo.py: Vehicle detection using YOLO classifier. It uses the YOLOv3 model to detect and classify vehicles in videos.

You Also Need to download the `yolov3.weights` , `coco.names` and `yolov3.cfg` files and put in the same directory as other code have to work.

# Requirements:
To run the scripts, you need to install the following Python libraries:

- OpenCV
- NumPy

  # Instructions:

Clone this repository to your local machine or download the main.py and yolo.py files.

Ensure that you have installed the required Python libraries mentioned in the requirements section.

If you want to run the Cascade Classifier method, simply execute the main.py script, passing the video file path as an argument:

```
python main.py
```

If you want to run the YOLO classifier method, follow these steps:

a. Download the YOLOv3 model files (yolov3.weights, yolov3.cfg, and coco.names) from the official YOLO website and place them in the same directory as yolo.py.

b. Execute the yolo.py script, passing the video file path as an argument:

```
python yolo.py 
```

# Notes:
- The Cascade Classifier method in main.py is simple but might not be as accurate in complex scenarios.
- The YOLO classifier method in yolo.py uses a more advanced deep learning model, providing better accuracy in vehicle detection.
- The YOLO classifier requires the model files to be downloaded separately, as mentioned in the requirements.
- The vehicle count is displayed in real-time on the video frames using a centroid-based tracking mechanism to avoid double-counting objects.
- You can adjust the confidence thresholds and other parameters in the scripts to optimize the performance according to your specific use case.
- Enjoy experimenting with the vehicle detection methods and analyzing different videos!






https://github.com/saqib772/Car-Detection-System/assets/121972215/37ab8d54-64ba-406f-85b6-444bc23594aa

