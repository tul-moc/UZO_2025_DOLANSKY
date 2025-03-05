import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

def plotFilteredImages(imDP1, ifft2_1, imDP2, ifft2_2):
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 2, 1)
    plt.title("Filter 1")
    plt.imshow(np.log(np.abs(imDP1)), cmap="jet")
    
    plt.subplot(2, 2, 2)
    plt.title("Filtered Image 1")
    plt.imshow(ifft2_1, cmap="gray")
    
    plt.subplot(2, 2, 3)
    plt.title("Filter 2")
    plt.imshow(np.log(np.abs(imDP2)), cmap="jet")
    
    plt.subplot(2, 2, 4)
    plt.title("Filtered Image 2")
    plt.imshow(ifft2_2, cmap="gray")
    
    plt.show()
    
def plotSpectrum(fft2, fft2_shift):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("FFT2")
    plt.imshow(np.log(np.abs(fft2)),cmap="jet")
    plt.subplot(1, 2, 2)
    plt.title("FFT2 Shifted")
    plt.imshow(np.log(np.abs(fft2_shift)),cmap="jet")
    plt.show()

def plotGrayAndDCT(gray_image, dctS):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Gray Image")
    plt.imshow(gray_image, cmap="gray")
    
    plt.subplot(1, 2, 2)
    plt.title("DCT Spectrum")
    plt.imshow(np.log(np.abs(dctS)), cmap="jet")
    
    plt.show()
    
def plotLimitedDCTs(limited_dct_10, idct_10,limited_dct_30,  idct_30,limited_dct_50, idct_50):
    
    plt.figure(figsize=(18, 12))
    plt.subplot(3, 2, 1)
    plt.title("DCT 10x10")
    plt.imshow(np.log(np.abs(limited_dct_10)), cmap="jet")

    plt.subplot(3, 2, 2)
    plt.title("IDCT 10x10")
    plt.imshow(idct_10, cmap="gray")

    plt.subplot(3, 2, 3)
    plt.title("DCT 30x30")
    plt.imshow(np.log(np.abs(limited_dct_30)), cmap="jet")

    plt.subplot(3, 2, 4)
    plt.title("IDCT 30x30")
    plt.imshow(idct_30, cmap="gray")

    plt.subplot(3, 2, 5)
    plt.title("DCT 50x50")
    plt.imshow(np.log(np.abs(limited_dct_50)), cmap="jet")

    plt.subplot(3, 2, 6)
    plt.title("IDCT 50x50")
    plt.imshow(idct_50, cmap="gray")

    plt.show()

def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def loadImage(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def convertFilt(dp_filtr, gray_image,fft2_shift):
    dp_filtr_gray = convertToGray(dp_filtr)
    dp_filtr_resized = cv2.resize(dp_filtr_gray, (gray_image.shape[1], gray_image.shape[0]))
    fft2_shift_filtered = fft2_shift * dp_filtr_resized
    return fft2_shift_filtered

def applyInverseFFT(imDP):
    ifft2 = np.fft.ifft2(np.fft.ifftshift(imDP))
    ifft2 = np.abs(ifft2)
    ifft2 = ifft2 / np.max(ifft2)
    return ifft2

def limitDCTCoefficients(dctS, size):
    limited_dct = np.zeros_like(dctS)
    limited_dct[:size, :size] = dctS[:size, :size]
    return limited_dct

def main():
    path_to_image = "cv04c_robotC.bmp"
    
    
    #-------------------TASK 1-------------------
    image = loadImage(path_to_image)
    gray_image = convertToGray(image)
    fft2 = np.fft.fft2(gray_image)
    fft2_shift = np.fft.fftshift(fft2)
    plotSpectrum(fft2,fft2_shift)
    
    
    #-------------------TASK 2-------------------
    dp_filtr1 = loadImage("cv04c_filtDP.bmp")
    dp_filtr2 = loadImage("cv04c_filtDP1.bmp")
    imDP1 = convertFilt(dp_filtr1, gray_image, fft2_shift)
    imDP2 = convertFilt(dp_filtr2, gray_image, fft2_shift)
    ifft2_1 = applyInverseFFT(imDP1)
    ifft2_2 = applyInverseFFT(imDP2)
    plotFilteredImages(imDP1, ifft2_1, imDP2, ifft2_2)
    
    hp_filder1 = loadImage("cv04c_filtHP.bmp")
    hp_filder2 = loadImage("cv04c_filtHP1.bmp")
    imHP1 = convertFilt(hp_filder1, gray_image, fft2_shift)
    imHP2 = convertFilt(hp_filder2, gray_image, fft2_shift)
    ifft2_hp1 = applyInverseFFT(imHP1)
    ifft2_hp2 = applyInverseFFT(imHP2)
    plotFilteredImages(imHP1, ifft2_hp1, imHP2, ifft2_hp2)
    
    #-------------------TASK 3-------------------
    dctS = dctn(gray_image)
    plotGrayAndDCT(gray_image, dctS)
    
    #-------------------TASK 4-------------------
    limited_dct_10 = limitDCTCoefficients(dctS, 10)
    limited_dct_30 = limitDCTCoefficients(dctS, 30)
    limited_dct_50 = limitDCTCoefficients(dctS, 50)

    idct_10 = idctn(limited_dct_10)
    idct_30 = idctn(limited_dct_30)
    idct_50 = idctn(limited_dct_50)
    plotLimitedDCTs(limited_dct_10, idct_10, limited_dct_30, idct_30, limited_dct_50, idct_50)
    
    
if __name__ == "__main__":
    main()