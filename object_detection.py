import cv2
import matplotlib.pyplot as plt
import numpy as np
model = cv2.dnn.readNetFromTensorflow('model/frozen_inference_graph.pb', 'model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# load the COCO class names
with open('model/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')
cap = cv2.VideoCapture(0)

while True:
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    
    ret , image = cap.read()
    if not ret :
        break
    image_height, image_width, _ = image.shape

    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
    output = model.forward()

    for detection in output[0,0,:,:]:
        score = float(detection[2])
        if score > 0.6:
            class_id = detection[1]
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]

            left = detection[3] * image_width
            top = detection[4] * image_height
            right = detection[5] * image_width
            bottom = detection[6] * image_height
            
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)
            cv2.putText(image, class_name, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            print(class_name, round(score*100,2),"%")
    cv2.imshow('image',image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()  