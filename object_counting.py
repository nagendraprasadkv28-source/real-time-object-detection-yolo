import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Store IDs of objects already counted
counted_ids = set()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Detect and track objects
    results = model.track(
        frame,
        persist=True,
        conf=0.50,
        verbose=False
    )

    for result in results:

        # Draw bounding boxes and labels
        frame = result.plot()

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:

            # Extract tracking IDs
            tracking_ids = boxes.id.int().cpu().tolist()

            # Remember each new ID
            for tracking_id in tracking_ids:
                counted_ids.add(tracking_id)

    # Total unique IDs observed
    total_count = len(counted_ids)

    # Display unique object count
    cv2.putText(
        frame,
        f"Unique Objects: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Unique Object Counting", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()