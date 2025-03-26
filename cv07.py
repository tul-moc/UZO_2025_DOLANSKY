import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def calculateGreenChannel(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = np.float32(img_rgb[:, :, 0]), np.float32(img_rgb[:, :, 1]), np.float32(img_rgb[:, :, 2])
    return 255 - ((G * 255) / (R + G + B))

def plotImage(img_orig, green_channel):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title('Originální obrázek')
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(132)
    plt.title('Zelená složka')
    plt.imshow(green_channel, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title('Histogram zelené složky')
    plt.hist(green_channel.ravel(), 256, [0, 256])
    plt.xlabel('Hodnota intenzity')
    plt.ylabel('Počet pixelů')
    plt.show()
    
def main():
    image_path = "cv07_segmentace.bmp"
    image = loadImage(image_path)
    green_channel = calculateGreenChannel(image)
    plotImage(image,green_channel)

if __name__ == "__main__":
    main()