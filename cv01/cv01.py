import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def load_image_list(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_list.append(os.path.join(directory, filename))
    return image_list

if __name__ == "__main__":
    directory_path = './img'
    images_path = load_image_list(directory_path)
    images = []
    histograms = []
    
    for image_path in images_path:
        img = cv.imread(image_path)
        if img is not None:
            resized_img = cv.resize(img, (50, 50))
            images.append((image_path, resized_img))
            histograms.append(cv.calcHist([resized_img], [0], None, [256], [0, 256]))
    
    sorted_images = []
    for i, (path, img) in enumerate(images):
        distances = []
        for j, hist in enumerate(histograms):
            if i != j:
                dist = cv.compareHist(histograms[i], hist, cv.HISTCMP_BHATTACHARYYA)
                distances.append((dist, images[j][1]))
        sorted_images.append((img, [x[1] for x in sorted(distances, key=lambda x: x[0])]))
    
    rows = len(images)
    cols = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for row, (orig_img, sorted_imgs) in enumerate(sorted_images):
        axes[row, 0].imshow(cv.cvtColor(orig_img, cv.COLOR_BGR2RGB))
        axes[row, 0].axis('off')
        
        for col, img in enumerate(sorted_imgs[:cols-1], start=1):
            axes[row, col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
