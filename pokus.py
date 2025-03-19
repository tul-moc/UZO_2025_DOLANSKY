import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def sobel_operator(image):
    """
    Implementace Sobelova operátoru pro detekci hran.
    
    Args:
        image: Vstupní šedotónový obraz
    
    Returns:
        magnitude: Magnitude gradientu (celková intenzita hran)
        gradient_x: Gradient ve směru x
        gradient_y: Gradient ve směru y
    """
    # Definice Sobelových jader
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    kernel_y = np.array([[1, 2, 1], 
                         [0, 0, 0], 
                         [-1, -2, -1]])
    
    # Aplikace konvoluce pro získání gradientů
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_y)
    
    # Výpočet magnitudy gradientu
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalizace magnitudy na rozsah 0-255
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    
    return magnitude, gradient_x, gradient_y

def compute_spectrum(image):
    """
    Výpočet frekvenčního spektra obrazu.
    
    Args:
        image: Vstupní obraz
    
    Returns:
        spectrum: Logaritmicky škálované spektrum
    """
    # Aplikace 2D Fourierovy transformace
    f = fftpack.fft2(image)
    
    # Přesun nulové frekvence do středu
    fshift = fftpack.fftshift(f)
    
    # Výpočet amplitudového spektra
    magnitude_spectrum = np.abs(fshift)
    
    # Logaritmické škálování pro lepší vizualizaci
    # Přidání malé hodnoty pro vyhnutí se log(0)
    spectrum = 20 * np.log10(magnitude_spectrum + 1e-10)
    
    return spectrum

def show_results(original, edges, spectrum):
    """
    Zobrazení výsledků detekce hran a spektra.
    
    Args:
        original: Původní obraz
        edges: Detekované hrany
        spectrum: Frekvenční spektrum
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Původní obrázek')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(edges, cmap='jet')
    plt.title('Sobelův hranový detektor')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(spectrum, cmap='jet')
    plt.title('Frekvenční spektrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Příklad použití
if __name__ == "__main__":
    # Načtení obrazu
    image = cv2.imread('cv06_robotC.bmp', cv2.IMREAD_GRAYSCALE)
    
    # Aplikace Sobelova operátoru
    edges, gradient_x, gradient_y = sobel_operator(image)
    
    # Výpočet spektra hran
    spectrum = compute_spectrum(edges)
    
    # Zobrazení výsledků
    show_results(image, edges, spectrum)