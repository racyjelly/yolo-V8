from ultralytics import YOLO

# Load a model
dir = "/mnt/sda/kjs_folder/yolo-V8/yolov8s.pt"
model = YOLO(dir)

data_dir = "/mnt/sda/kjs_folder/yolo-V8/ultralytics/yolo/data/datasets/military.yaml"

# Train the model
results = model.train(
    data=data_dir,
    epochs=100,
    batch=64)

# Evaluate model performance on the validation set
metrics = model.val()

results = model("/mnt/sda/kjs_folder/Military_datasets/data/Validation/images/I1_S0_C5_0001008.jpg")
results[0].save("test_output.jpg")

success = model.export(format="onnx")