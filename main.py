import os
import cv2
import torch
import matplotlib.pyplot as plt
import pytesseract

# Load the pre-trained YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

# Define the path to your test dataset
test_dataset_path = "D:/Kuliah/zhuanti/test_dir"

# Function to detect license plates and recognize text using YOLOv5 and Tesseract OCR
def detect_and_recognize_license_plates(image_path):
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    results = model(img)
    print("Inference completed.")
    
    # Ensure results are valid
    if not results:
        print("No results returned from model inference.")
        return
    
    # Process the results
    detections = results.pandas().xyxy[0]
    print("Detections processed.")
    print(detections)  # Print the detection results for debugging
    
    # Define the class index for 'license plate', assuming it's class 2
    # Note: You need to verify the correct class index for license plates in your model
    license_plate_class = 2
    
    for _, row in detections.iterrows():
        if row['class'] == license_plate_class:
            print(f"License plate detected with confidence {row['confidence']:.2f}")
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Extract the detected license plate region
            plate_img = img[y1:y2, x1:x2]
            
            # Perform OCR on the detected license plate region
            plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')
            print(f"Detected license plate text: {plate_text}")
            
            # Draw bounding box and label on the original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    show_image(img)

# Function to display the image using matplotlib
def show_image(img):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.close()  # Close the figure after displaying

# Get a list of all image files in the test dataset directory
test_image_files = [os.path.join(test_dataset_path, f) for f in os.listdir(test_dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Check if test images are found
if not test_image_files:
    print("No images found in the test dataset directory.")

# Process images from the test dataset
for image_path in test_image_files:
    detect_and_recognize_license_plates(image_path)

print("Processing completed.")
