import cv2
import numpy as np
import matplotlib.pyplot as plt


def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image


def load_images_matrix(pattern_format="p{}{}.bmp", person_range=(1, 3), image_range=(1, 3)):
    images_matrix = []
    for person_id in range(person_range[0], person_range[1] + 1):
        row = []
        for img_id in range(image_range[0], image_range[1] + 1):
            img = cv2.imread(pattern_format.format(person_id, img_id), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Image p{person_id}{img_id}.bmp could not be loaded")
            img = cv2.resize(img, (64, 64))
            img = img.flatten()
            row.append(img)
        images_matrix.append(row)
    return images_matrix


def convertToGray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def create_training_data(images_matrix):
    training_data = []
    for person_images in images_matrix:
        for img in person_images:
            training_data.append(img)
    Wp = np.array(training_data).T
    wp = np.mean(Wp, axis=1).reshape(-1, 1)
    W = Wp - wp
    return Wp, wp, W


def calculate_eigenspace(W):
    C = np.dot(W.T, W)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    Ep = eigenvectors[:, idx]
    E = np.dot(W, Ep)
    for i in range(E.shape[1]):
        E[:, i] /= np.linalg.norm(E[:, i])
    return E, eigenvalues, Ep


def project_images(E, W):
    return np.dot(E.T, W)


def recognize_face(E, wp, PI, unknown_image):
    unknown_gray = convertToGray(unknown_image)
    unknown_gray = cv2.resize(unknown_gray, (64, 64))
    wpu = unknown_gray.flatten()
    wu = wpu - wp.flatten()
    PT = np.dot(E.T, wu)
    distances = [np.linalg.norm(PI[:, i] - PT) for i in range(PI.shape[1])]
    min_distance_idx = np.argmin(distances)
    person_id = min_distance_idx // 3 + 1
    image_id = min_distance_idx % 3 + 1
    return person_id, image_id, min_distance_idx, distances


def display_results(unknown_image, recognized_image, person_id, image_id):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(convertToGray(unknown_image), cmap='gray')
    plt.title("Unknown Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(convertToGray(recognized_image), cmap='gray')
    plt.title(f"Recognized as: Person {person_id}, Image {image_id}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('pca_recognition_result.png')
    plt.show()


def main():
    try:
        unknown_image = loadImage('unknown.bmp')
        images_matrix = load_images_matrix()
        Wp, wp, W = create_training_data(images_matrix)
        E, eigenvalues, Ep = calculate_eigenspace(W)
        PI = project_images(E, W)
        person_id, image_id, min_distance_idx, distances = recognize_face(E, wp, PI, unknown_image)
        recognized_image = loadImage(f"p{person_id}{image_id}.bmp")
        display_results(unknown_image, recognized_image, person_id, image_id)
        print(f"The unknown image was recognized as: Person {person_id}, Image {image_id}")
        print(f"Euclidean distance: {distances[min_distance_idx]}")
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
