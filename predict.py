import torch
from torchvision import transforms as T
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_frcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_ssdlite_model(num_classes):
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    # Note: The model's classification head is not replaced for custom class count
    # Adjust if using custom training from scratch
    return model

def load_model(model_name, weights_path, num_classes):
    if model_name == "fasterrcnn":
        model = get_frcnn_model(num_classes)
    elif model_name == "ssdlite":
        model = get_ssdlite_model(num_classes)
    else:
        raise ValueError("Invalid model name. Use 'fasterrcnn' or 'ssdlite'.")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def visualize_prediction(image, prediction, class_names, score_threshold=0.5):
    image = image.copy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label_text = f"{class_names.get(str(label), str(label))} ({score:.2f})"
        plt.text(x1, y1 - 10, label_text, color='white', 
                 bbox=dict(facecolor='red', alpha=0.5))

    plt.axis("off")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["fasterrcnn", "ssdlite"], help="Model to use")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--classes", required=True, help="Path to classes.json file")
    args = parser.parse_args()

    # Load class names
    with open(args.classes) as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    model = load_model(args.model, args.weights, num_classes)

    # Load and transform image
    image = Image.open(args.image).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image).to(DEVICE)

    # Predict
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # Visualize
    visualize_prediction(image, prediction, class_names)

if __name__ == "__main__":
    main()
