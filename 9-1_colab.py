import numpy as np
import cv2 as cv
import sys
from google.colab.patches import cv2_imshow

def construct_yolo_v3(weights_path, config_path, names_path):
    """Construct the YOLO model with the given file paths."""
    with open(names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    model = cv.dnn.readNet(weights_path, config_path)
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):
    """Detect objects in the image using YOLO."""
    height, width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)

    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    box, conf, id = [], [], []  # Boxes, confidences, and class IDs
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects

# Define file paths
weights_path = '/content/yolov3.weights'
config_path = '/content/yolov3.cfg'
names_path = '/content/coco_names.txt'
image_path = '/content/soccer.jpg'

# Ensure files exist
import os
for file_path in [weights_path, config_path, names_path, image_path]:
    if not os.path.exists(file_path):
        sys.exit(f"Error: {file_path} not found. Please upload the file to /content folder.")

# Construct YOLO model
model, out_layers, class_names = construct_yolo_v3(weights_path, config_path, names_path)
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Read input image
img = cv.imread(image_path)
if img is None:
    sys.exit("Error: Unable to read the image file.")

# Detect objects
res = yolo_detect(img, model, out_layers)

# Draw results on the image
for i in range(len(res)):
    x1, y1, x2, y2, confidence, id = res[i]
    text = str(class_names[id]) + ' %.3f' % confidence
    cv.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
    cv.putText(img, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

# Display the image
cv2_imshow(img)
