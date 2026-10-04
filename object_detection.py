import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

previous_positions = {}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    current_positions = {}

    for result in results:
        for box in result.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            # Center of object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display object name
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Store position
            current_positions[class_name] = (center_x, center_y)

            # Check movement
            if class_name in previous_positions:

                old_x, old_y = previous_positions[class_name]

                distance = ((center_x - old_x) ** 2 +
                            (center_y - old_y) ** 2) ** 0.5

                if distance > 10:
                    status = "MOVING"
                else:
                    status = "STATIONARY"

                cv2.putText(
                    frame,
                    status,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    previous_positions = current_positions

    # Show frame
    cv2.imshow("YOLO Object Detection & Movement Analysis", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()