import os
# Verify that the dataset and data.yaml file exist in your Google Drive
train_path = "/content/drive/MyDrive/dataset/images/train"
val_path = "/content/drive/MyDrive/dataset/images/val"
yaml_path = "/content/drive/MyDrive/dataset/data.yaml"
assert os.path.exists(train_path), f"Train images path not found: {train_path}"
assert os.path.exists(val_path), f"Validation images path not found: {val_path}"
assert os.path.exists(yaml_path), f"YAML file not found: {yaml_path}"
print("All paths verified successfully!")
!pip install ultralytics
from ultralytics import YOLO
# Load the YOLO model (YOLOv5 or YOLOv8)
model = YOLO("yolov5nu.pt")  # You can also use yolov8n.pt for YOLOv8
# Train the model using your data.yaml file
model.train(
    data=yaml_path,  # Path to your YAML file
    epochs=50,       # Number of epochs
    imgsz=640,       # Image size
    batch=16,        # Batch size
    device=0         # Use 0 for GPU or 'cpu' for CPU
)
metrics = model.val()
print("Validation metrics:", metrics)
test_image_path = "/content/signature-6-_jpg.rf.549cdbfcc6811e9be3cd0abbfcee3d5b.jpg"  # Replace with an actual test image path
if os.path.exists(test_image_path):
    results = model(test_image_path)  # Run inference on the test image
    results[0].show()  # Visualize the results
else:
    print(f"Test image not found: {test_image_path}")
export_path = model.export(format="onnx")  # Export the model to ONNX format
print(f"Model exported to: {export_path}")