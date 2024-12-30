from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("yolov8x-world.pt")  # or choose yolov8m/l-world.pt
model.export(format="onnx", simplify=True, opset=13, nms=False, optimize=True, imgsz=640, dynamic=True)
# Define custom classes
model.set_classes(["person", "bus"])
model = YOLO("yolov8x-world.onnx")

# Execute prediction for specified categories on an image
results = model.predict("/storage/SSD-3/yptsai/data/coco2017/val2017/000000127270.jpg")

# Show results
results[0].show()