import numpy as np
import cv2

# Paths to model and configuration
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

# Define the class labels
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Assign random colors to classes
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width = image.shape[:2]

    # Prepare input blob and perform forward pass
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    # Loop through detections
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {confidence * 100:.2f}%"

            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y),
                          colors[class_index], 2)
            cv2.putText(image, prediction_text, (upper_left_x, upper_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    # Display the output
    cv2.imshow("Detected Objects", image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
