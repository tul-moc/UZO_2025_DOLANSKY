import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def convertToGray(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray

def topHatMethod(image, kernel_size=10):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernel_size+1, 2*kernel_size+1))
    eroded = cv2.erode(image, se)
    opened = cv2.dilate(eroded, se)
    tophat_image = cv2.subtract(image, opened)
    return tophat_image

def segmentImage(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def countObjects(thresh, min_size=90):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    count = 0
    valid_centroids = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            count += 1
            valid_centroids.append(centroids[i])
    return count, valid_centroids

def plotResults(image_gray, tophat_image, original_thresh, tophat_thresh):
    plt.figure(figsize=(12, 10))
    plt.subplot(321)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Původní obraz')
    plt.subplot(322)
    plt.hist(image_gray.ravel(), 256, [0, 256])
    plt.title('Histogram původního obrazu')
    plt.subplot(323)
    plt.imshow(original_thresh, cmap='gray')
    plt.title('Segmentovaný původní obraz')
    plt.subplot(324)
    plt.imshow(tophat_image, cmap='gray')
    plt.title('Top-hat upravený obraz')
    plt.subplot(325)
    plt.hist(tophat_image.ravel(), 256, [0, 256])
    plt.title('Histogram top-hat obrazu')
    plt.subplot(326)
    plt.imshow(tophat_thresh, cmap='gray')
    plt.title('Segmentovaný top-hat obraz')
    plt.tight_layout()
    plt.savefig('segmentace.png')
    plt.show()

def plotCentroids(image, centroids, title, output_file):
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    for x, y in centroids:
        plt.plot(x, y, 'r*', markersize=5)
    plt.savefig(output_file)
    plt.show()

def main():
    try:
        image = loadImage('cv09_rice.bmp')
        image_gray = convertToGray(image)
    except FileNotFoundError as e:
        print(e)
        return

    tophat_image = topHatMethod(image_gray, kernel_size=10)
    original_thresh = segmentImage(image_gray)
    tophat_thresh = segmentImage(tophat_image)

    original_count, original_centroids = countObjects(original_thresh)
    tophat_count, tophat_centroids = countObjects(tophat_thresh)

    print(f"Počet zrníček rýže na původním obrazku: {original_count}")
    print(f"Počet zrníček rýže na upraveném obrazku (top-hat): {tophat_count}")

    plotResults(image_gray, tophat_image, original_thresh, tophat_thresh)

if __name__ == "__main__":
    main()
