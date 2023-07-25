# Object Detection with SSD MobileNet V2

This code is an implementation of real-time object detection using the SSD (Single Shot Multibox Detector) with MobileNet V2 as the base network. It uses OpenCV's Deep Neural Network (DNN) module for inference and the COCO dataset's pre-trained model for detection.

## Prerequisites

Before running this code, make sure you have the following:

1. Python installed on your system.
2. OpenCV (`cv2`) and Matplotlib (`matplotlib`) libraries.
3. The pre-trained SSD MobileNet V2 model (`frozen_inference_graph.pb`) and its corresponding configuration file (`ssd_mobilenet_v2_coco_2018_03_29.pbtxt`).
4. The COCO class names file (`object_detection_classes_coco.txt`).

## Installation

1. Install Python by downloading it from the official website and following the installation instructions for your operating system.

2. Install required libraries using pip:
   ```
   pip install opencv-python matplotlib
   ```

## Usage

1. Place the pre-trained model files (`frozen_inference_graph.pb` and `ssd_mobilenet_v2_coco_2018_03_29.pbtxt`) in the `model/` directory.

2. Download the COCO class names file (`object_detection_classes_coco.txt`) and place it in the `model/` directory.

3. Run the script using the following command:
   ```
   python object_detection.py
   ```

## How it works

1. The script loads the SSD MobileNet V2 model and the COCO class names from the files provided.

2. It initializes the webcam (or any other video source with appropriate changes) using OpenCV (`cv2.VideoCapture(0)`).

3. For each frame from the video stream, it performs the following steps:
   - Preprocess the frame by converting it into a blob with a size of (300, 300).
   - Forward the blob through the pre-trained model to get detection results.
   - Iterate through the detections and filter out objects with scores greater than 0.6 (adjustable).
   - Draw bounding boxes and labels on the frame for the detected objects.

4. The processed frame is displayed in a window until the user presses the 'Esc' key to exit the program.

## Customization

You can customize the following parameters according to your requirements:

- `model/frozen_inference_graph.pb`: You can use other pre-trained models compatible with TensorFlow's `pb` format.
- `model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt`: Configuration file for the chosen pre-trained model.
- `model/object_detection_classes_coco.txt`: Class names corresponding to the COCO dataset.
- `score > 0.6`: The confidence threshold (score) for object detection. You can modify it to get more or fewer detections.
- `cv2.VideoCapture(0)`: If you want to use a video file instead of the webcam, replace `0` with the path to your video file.

Feel free to experiment with different models, classes, and confidence thresholds to suit your specific needs.

**Note:** Ensure that you have the necessary permissions to use and distribute the pre-trained model files and the COCO class names file in your application.

## Acknowledgments

This code is based on the SSD MobileNet V2 implementation available in OpenCV's DNN module and utilizes the COCO dataset for object classes. We acknowledge the authors of these models and datasets for their valuable contributions to the computer vision community.