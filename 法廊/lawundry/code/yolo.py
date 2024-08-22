import argparse
import cv2
import os
import logging
from PIL import Image, ImageFilter
from ultralytics import YOLO

def process_image(input_path, output_txt, partpic_dir, image_index):
    # Disable logging from ultralytics
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    # Load YOLOv10 model
    model = YOLO('yolov10x.pt')

    # Load image
    img_cv2 = cv2.imread(input_path)
    if img_cv2 is None:
        raise FileNotFoundError(f"Cannot load image file: {input_path}")

    # Resize image
    scale_percent = 200
    width = int(img_cv2.shape[1] * scale_percent / 100)
    height = int(img_cv2.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_cv2 = cv2.resize(img_cv2, dim, interpolation=cv2.INTER_LINEAR)

    # Convert image to PIL
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

    # Perform detection
    results = model(img_pil)

    # Restore logging config (if needed)
    logging.getLogger('ultralytics').setLevel(logging.INFO)

    # Get detection results
    detections = results[0].boxes.data.cpu().numpy()

    # Prepare output directory
    os.makedirs(partpic_dir, exist_ok=True)

    # Initialize counters
    object_counts = {}
    numOfAll = 0
    detected_objects = []

    # Process detections
    with open(output_txt, 'w') as file:
        for idx, (*box, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # Update object counts
            class_name = model.names[int(cls)].replace(" ", "_")
            numOfAll += 1
            detected_objects.append(class_name)
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1

            # Draw bounding box and label
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save cropped images
            cropped_img = img_cv2[y1:y2, x1:x2]
            cropped_img_path = os.path.join(partpic_dir, f"detection_img{image_index}_{idx}_{class_name}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)

            # Sharpen cropped images
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            sharp_img = cropped_img_pil.filter(ImageFilter.UnsharpMask(radius=5, percent=100, threshold=10))
            sharp_img_path = os.path.join(partpic_dir, f"sharp_img{image_index}_{idx}_{class_name}.jpg")
            sharp_img.save(sharp_img_path)

        # Print results
        file.write(f"{numOfAll}\n")
        for obj in detected_objects:
            file.write(f"{obj}\n")

def main(input_path1, input_path2):
    # Define output paths
    output_txt1 = "YOLOresult1.txt"
    output_txt2 = "YOLOresult2.txt"
    partpic_dir = "static/partpic"

    # Process both images
    process_image(input_path1, output_txt1, partpic_dir, image_index=1)
    process_image(input_path2, output_txt2, partpic_dir, image_index=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv10x Object Detection')
    parser.add_argument('input_file1', type=str, help='Path to the first input image file')
    parser.add_argument('input_file2', type=str, help='Path to the second input image file')
    args = parser.parse_args()

    main(args.input_file1, args.input_file2)