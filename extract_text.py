import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Set Tesseract-OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image = cv2.imread('sample.jpeg')

# Preprocessing: Convert to grayscale and apply Gaussian blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding for better text extraction
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Extract text using Tesseract
extracted_text = pytesseract.image_to_string(thresh)

# Print extracted text
print("Extracted Text:\n", extracted_text)

# Function to organize text into a dictionary with enhanced structuring
def organize_text(text):
    organized_dict = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        # Basic keyword detection for categories
        if ":" in line:
            key, value = line.split(":", 1)
            organized_dict[key.strip()] = value.strip()
        else:
            # Treat any non-key-value line as a potential key
            # You can define specific logic to classify these lines as needed
            organized_dict[line] = ""

    return organized_dict

# Organize the extracted text
organized_dict = organize_text(extracted_text)

# Print organized dictionary with improved formatting
print("Organized Dictionary:")
for key, value in organized_dict.items():
    print(f"{key}: {value}")

# Visualize the processed image with contours for better understanding
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with contours
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Processed Image with Detected Text Regions')
plt.axis('off')
plt.show()

# Print organized dictionary
print("Organized Dictionary:\n", organized_dict)
