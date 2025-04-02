import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def thresholdImage(channel,threshold):
    _, binary = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def display_results(original, binary, title="Segmentace"):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Původní obrázek")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary, cmap='gray')
    plt.title(f"title")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    image1_path = "cv08_im1.bmp"
    image1 = loadImage(image1_path)
    
    # Stanovení prahu a segmentace prvního obrázku
    threshold = 120  # Můžete experimentovat s různými hodnotami
    image_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    binary1 = thresholdImage(image_gray, threshold)
    
    binary1 = cv2.bitwise_not(binary1)
    
    display_results(image1, binary1, f"Segmentace obrázku 1 (práh = {threshold})")


if __name__ == "__main__":
    main()