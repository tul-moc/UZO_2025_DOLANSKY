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

def threshold_image(green_channel):
    _, binary = cv2.threshold(green_channel.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def customConnectedComponents(binary_image):
    h, w = binary_image.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    coins = []
    
    def dfs(i, j):
        stack = [(i, j)]
        pixels = []
        while stack:
            ci, cj = stack.pop()
            if labels[ci, cj] != 0:
                continue
            labels[ci, cj] = current_label
            pixels.append((ci, cj))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if binary_image[ni, nj] == 255 and labels[ni, nj] == 0:
                        stack.append((ni, nj))
        return pixels

    for i in range(h):
        for j in range(w):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                pixels = dfs(i, j)
                pixels_count = len(pixels)
                centroid_x = sum(j for i, j in pixels) / pixels_count
                centroid_y = sum(i for i, j in pixels) / pixels_count
                coin_value = "5" if pixels_count > 4000 else "1"
                coins.append({
                    'centroid': (centroid_x, centroid_y),
                    'type': coin_value
                })
                current_label += 1
    return labels, coins


def drawCentroids(image, coins):
    centroids = image.copy()
    total_value = 0
    for coin in coins:
        x, y = map(int, coin['centroid'])
        cv2.circle(centroids, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(centroids, f"{coin['type']} Kč", (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        total_value += int(coin['type'])
        print(f"Na souřadnici ({x}, {y}) je mince {coin['type']} CZK")
    return centroids, total_value

def plotResults(img_orig, green_channel, labels, img_with_centroids, total_value):
    plt.figure(figsize=(16, 4))
    plt.subplot(141)
    plt.title('Originální obrázek')
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(142)
    plt.title('Zelená složka')
    plt.imshow(green_channel, cmap='gray')
    plt.axis('off')
    plt.subplot(143)
    plt.title('Označené oblasti')
    plt.imshow(labels, cmap='nipy_spectral')
    plt.axis('off')
    plt.subplot(144)
    plt.title('Detekované mince')
    plt.imshow(cv2.cvtColor(img_with_centroids, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Celková hodnota mincí: {total_value} CZK")

def main():
    image_path = "cv07_segmentace.bmp"
    image = loadImage(image_path)
    green_channel = calculateGreenChannel(image)
    binary = threshold_image(green_channel)
    labels, coins = customConnectedComponents(binary)
    centroids, total_value = drawCentroids(image, coins)
    plotResults(image, green_channel, labels, centroids, total_value)

if __name__ == "__main__":
    main()