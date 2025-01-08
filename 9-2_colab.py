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
video_path = '/content/video.mp4'  # Replace with your uploaded video file

# Ensure files exist
import os
for file_path in [weights_path, config_path, names_path, video_path]:
    if not os.path.exists(file_path):
        sys.exit(f"Error: {file_path} not found. Please upload the file to /content folder.")

# Construct YOLO model
model, out_layers, class_names = construct_yolo_v3(weights_path, config_path, names_path)
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Open video file
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit("Error: Unable to open the video file.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to fetch frame.")
        break

    # Detect objects
    res = yolo_detect(frame, model, out_layers)

    # Draw results on the frame
    for i in range(len(res)):
        x1, y1, x2, y2, confidence, id = res[i]
        text = str(class_names[id]) + ' %.3f' % confidence
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[id], 2)
        cv.putText(frame, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

    # Display the frame (in Colab, use cv2_imshow)
    cv2_imshow(frame)

    # Simulate a real-time effect with a small delay (use a reasonable delay for Colab)
    key = cv.waitKey(1)
    if key == ord('q'):  # Quit if 'q' is pressed
        break

cap.release()  # Release the video capture
cv.destroyAllWindows()
