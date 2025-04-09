import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def convertToGray(image):
    if len(image.shape) == 3:  # Barevný obrázek
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Už je šedotónový
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

def plotResults(image_gray, tophat_image, original_thresh, tophat_thresh, image_with_centroids, tophat_count):
    fig = plt.figure(figsize=(10, 14))
    
    # Histogram původního obrazu
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.hist(image_gray.ravel(), 256, [0, 256], color='steelblue')
    ax1.set_title('hist. orig. image')
    
    # Histogram top-hat obrazu
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.hist(tophat_image.ravel(), 256, [0, 256], color='steelblue')
    ax2.set_title('hist. top-hat image')

    # Segmentovaný původní obraz
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.imshow(original_thresh, cmap='gray')
    ax3.set_title('seg. orig. image')
    ax3.axis('off')

    # Segmentovaný top-hat obraz
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.imshow(tophat_thresh, cmap='gray')
    ax4.set_title('seg. top-hat image')
    ax4.axis('off')

    # Text: počet zrníček
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.imshow(image_with_centroids, cmap='gray')
    ax5.axis('off')
    ax5.set_title(f'Number of rice grains: {tophat_count}', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('segmentace_s_centroidy.png')
    plt.show()

    
    
def drawCentroids(image, centroids):
    image_with_centroids = image.copy()
    for x, y in centroids:
        x, y = int(round(x)), int(round(y))
        cv2.drawMarker(image_with_centroids, (x, y), (0, 255, 0), 
                       markerType=cv2.MARKER_CROSS, markerSize=8, 
                       thickness=1, line_type=cv2.LINE_AA)
    return image_with_centroids


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

    image_with_centroids = drawCentroids(image_gray, tophat_centroids)


    print(f"Počet zrníček rýže na původním obrazku: {original_count}")
    print(f"Počet zrníček rýže na upraveném obrazku (top-hat): {tophat_count}")
    plotResults(image_gray, tophat_image, original_thresh, tophat_thresh, image_with_centroids, tophat_count)

if __name__ == "__main__":
    main()
