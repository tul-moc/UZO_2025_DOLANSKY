import cv2
import numpy as np
import matplotlib.pyplot as plt

def lapcer(gray_image):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    image_lap = cv2.filter2D(gray_image, cv2.CV_16S, kernel=kernel)
    return image_lap

def sobel(image):
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    kernel_y = np.array([[1, 2, 1], 
                         [0, 0, 0], 
                         [-1, -2, -1]])
    
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_y)
    
    image_sob = np.sqrt(gradient_x**2 + gradient_y**2)
    
    image_sob = np.uint8(255 * image_sob / np.max(image_sob))
    
    return image_sob

def kirs(image):
    kernel_x = np.array([[-5, 3, 3], 
                         [-5, 0, 3], 
                         [-5, 3, 3]])
    
    kernel_y = np.array([[3,3, 3], 
                         [3, 0, 3], 
                         [-5, -5, -5]])
    
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_y)
    
    image_sob = np.sqrt(gradient_x**2 + gradient_y**2)
    
    image_sob = np.uint8(255 * image_sob / np.max(image_sob))
    
    return image_sob

def lapcer_test(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernels = [
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
        np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]),
        np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    ]
    results = []
    for kernel in kernels:
        image_lap = cv2.filter2D(gray_image, cv2.CV_16S, kernel=kernel)
        results.append(image_lap)
    return results

def display_laplace_results(image, laplace_images):
    plt.figure(figsize=(10, 8))
    for i, laplace_image in enumerate(laplace_images):
        plt.subplot(2, 2, i + 1)
        plt.title(f"Kernel {i + 1}")
        plt.imshow(compute_spectrum(laplace_image), cmap="jet")
        plt.colorbar(label="Intensity")
    
    plt.tight_layout()
    plt.show()

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image
def compute_spectrum(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))))
    return spectrum

def display_results(image, original_spectrum, image_lap, laplace_spectrum):
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    
    plt.subplot(2, 2, 2)
    plt.title("Spectrum of Original Image")
    plt.imshow(original_spectrum, cmap="jet")
    plt.colorbar(label="Intensity")
    
    plt.subplot(2, 2, 3)
    plt.title("Laplace Filtered Image")
    plt.imshow(image_lap, cmap="jet")
    plt.colorbar(label="Intensity")
    
    plt.subplot(2, 2, 4)
    plt.title("Spectrum of Laplace Filtered Image")
    plt.imshow(laplace_spectrum, cmap="jet")
    plt.colorbar(label="Intensity")
    
    plt.tight_layout()
    plt.show()
    
def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def main():
    image_path = "cv04c_robotC.bmp"
    image = loadImage(image_path)
    gray_image = convertToGray(image)
    image_lap = lapcer(gray_image)
    image_sob = sobel(gray_image)
    image_kir = kirs(gray_image)
    
    original_spectrum = compute_spectrum(image)
    display_results(gray_image, original_spectrum, image_lap, compute_spectrum(image_lap))
    display_results(gray_image, original_spectrum, image_sob, compute_spectrum(image_sob))
    display_results(gray_image, original_spectrum, image_kir, compute_spectrum(image_kir))

if __name__ == "__main__":
    main()