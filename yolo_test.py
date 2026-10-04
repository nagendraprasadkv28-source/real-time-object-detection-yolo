from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in image
results = model("input.jpg", show=True)

# Print detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f}")