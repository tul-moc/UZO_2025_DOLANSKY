import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv09 import topHatMethod

def main():
    # Načtení původního obrazu
    image = cv2.imread('cv09_rice.bmp', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Nepodařilo se načíst obraz. Ujistěte se, že soubor 'cv09_rice.bmp' existuje ve správném umístění.")
        return
    
    # 1. Top-hat transformace
    # Vytvoření strukturního elementu (disk)
    se_size = 10
    tophat_image = topHatMethod(image, se_size)
    
    # 2. Segmentace obrazů pomocí prahování
    # Nalezení vhodných prahů pro původní a upravený obraz
    _, original_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tophat_thresh = cv2.threshold(tophat_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Identifikace objektů (zrníček) pomocí barvení oblastí pomocí cv2.connectedComponentsWithStats
    num_labels_original, labels_original, stats_original, centroids_original = cv2.connectedComponentsWithStats(original_thresh)
    num_labels_tophat, labels_tophat, stats_tophat, centroids_tophat = cv2.connectedComponentsWithStats(tophat_thresh)
    
    # 4. Počítání zrníček s omezením minimální velikosti (90 pixelů)
    min_size = 90
    original_centroids = []
    tophat_centroids = []
    original_count = 0
    tophat_count = 0
    
    # První index (0) odpovídá pozadí, proto začínáme od 1
    for i in range(1, num_labels_original):
        if stats_original[i, cv2.CC_STAT_AREA] >= min_size:
            original_count += 1
            original_centroids.append((centroids_original[i][1], centroids_original[i][0]))  # (y, x) pro plt.plot
    
    for i in range(1, num_labels_tophat):
        if stats_tophat[i, cv2.CC_STAT_AREA] >= min_size:
            tophat_count += 1
            tophat_centroids.append((centroids_tophat[i][1], centroids_tophat[i][0]))  # (y, x) pro plt.plot
    
    # Výpis počtu zrníček
    print(f"Počet zrníček rýže na původním obrazku: {original_count}")
    print(f"Počet zrníček rýže na upraveném obrazku (top-hat): {tophat_count}")
    
    # Zobrazení výsledků v jednom okně
    plt.figure(figsize=(12, 10))
    
    # Původní obraz
    plt.subplot(321)
    plt.imshow(image, cmap='gray')
    plt.title('Původní obraz')
    
    # Histogram původního obrazu
    plt.subplot(322)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title('Histogram původního obrazu')
    
    # Segmentovaný původní obraz
    plt.subplot(323)
    plt.imshow(original_thresh, cmap='gray')
    plt.title('Segmentovaný původní obraz')
    
    # Top-hat upravený obraz
    plt.subplot(324)
    plt.imshow(tophat_image, cmap='gray')
    plt.title('Top-hat upravený obraz')
    
    # Histogram upraveného obrazu
    plt.subplot(325)
    plt.hist(tophat_image.ravel(), 256, [0, 256])
    plt.title('Histogram top-hat obrazu')
    
    # Segmentovaný upravený obraz
    plt.subplot(326)
    plt.imshow(tophat_thresh, cmap='gray')
    plt.title('Segmentovaný top-hat obraz')
    
    plt.tight_layout()
    plt.savefig('segmentace.png')
    plt.show()
    
    # Zobrazení těžišť v původním obrazu
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Původní obraz s vyznačenými těžišti zrníček')
    
    # Vykreslení těžišť původního obrazu
    for y, x in original_centroids:
        plt.plot(x, y, 'r*', markersize=5)
    
    plt.savefig('teziste.png')
    plt.show()
    
    # Zobrazení těžišť v upraveném obrazu
    plt.figure(figsize=(10, 8))
    plt.imshow(tophat_image, cmap='gray')
    plt.title('Top-hat obraz s vyznačenými těžišti zrníček')
    
    # Vykreslení těžišť upraveného obrazu
    for y, x in tophat_centroids:
        plt.plot(x, y, 'r*', markersize=5)
    
    plt.savefig('teziste_tophat.png')
    plt.show()

if __name__ == "__main__":
    main()