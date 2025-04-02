import cv2
import numpy as np
import matplotlib.pyplot as plt
# použít izomorfní transformace pro 1., pro druhou ne

def convertToGray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    return inverted

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def calculateRedChannel(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = np.float32(img_rgb[:, :, 0]), np.float32(img_rgb[:, :, 1]), np.float32(img_rgb[:, :, 2])
    red_channel = 255 - ((R * 255) / (R + G + B))
    inverted_red_channel = cv2.bitwise_not(red_channel.astype(np.uint8))
    return inverted_red_channel

def thresholdImage(channel,threshold):
    _, binary = cv2.threshold(channel.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def colorRegions(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    labels_filtered = labels.copy()
    coins_info = []
    for i in range(1, num_labels):
        coins_info.append({
            'label': i,
            'area': stats[i, cv2.CC_STAT_AREA],
            'centroid': centroids[i]
        })
    return labels_filtered, coins_info

def plotImgs1(original, gray, filtered, originalX):
    plt.figure(figsize=(16, 4))
    
    plt.subplot(141)
    plt.title('Originální obrázek')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(142)
    plt.title('Šedá složka')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(143)
    plt.title('Po filtru')
    plt.imshow(filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(144)
    plt.title('Result')
    plt.imshow(originalX)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def plotImgs2(original, gray, filtered, originalX):
    plt.figure(figsize=(16, 4))
    
    plt.subplot(141)
    plt.title('Originální obrázek')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(142)
    plt.title('Šedá složka')
    plt.imshow(gray, cmap='jet')
    plt.axis('off')

    plt.subplot(143)
    plt.title('Po filtru')
    plt.imshow(filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(144)
    plt.title('Result')
    plt.imshow(cv2.cvtColor(originalX, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()    

def drawCentroids(image, coins_info):
    image_with_centroids = image.copy()
    for coin in coins_info:
        x, y = map(int, coin['centroid'])
        cv2.drawMarker(image_with_centroids, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, 
                       markerSize=8, thickness=1, line_type=cv2.LINE_AA)
    return image_with_centroids

def openMethod(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    opened_image = (opened_image > 0).astype(np.uint8)
    return opened_image

def closeMethod(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    closed_image = (closed_image > 0).astype(np.uint8)
    return closed_image


def main():
    image1 = loadImage("cv08_im1.bmp")
    image1_gray = convertToGray(image1)
    image1_thresh = thresholdImage(image1_gray,104)
    binary_im1 = openMethod(image1_thresh, 8)
    _, info_im1 = colorRegions(binary_im1)
    img1_X = drawCentroids(image1, info_im1)
    # plotImgs1(image1,image1_thresh,binary_im1,img1_X)
    
    image2 = loadImage("cv08_im2.bmp")
    image2_red = calculateRedChannel(image2)
    image2_thresh = thresholdImage(image2_red, 104)
    binary_im2 = closeMethod(image2_thresh, 8)
    _, info_im2 = colorRegions(binary_im2)
    img2_X = drawCentroids(image2, info_im2)
    plotImgs2(image2, image2_red, image2_thresh, img2_X)
    
    
if __name__ == "__main__":
    main()