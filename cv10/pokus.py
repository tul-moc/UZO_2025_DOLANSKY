import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_images(pattern_format="p{}{}.bmp", person_range=(1, 3), image_range=(1, 3), show_info=True):
    """
    Load images based on pattern with proper organization.
    Each person has multiple images: p11, p12, p13 belongs to person 1, etc.
    """
    images = []
    filenames = []
    person_labels = []
    
    for person_id in range(person_range[0], person_range[1] + 1):
        person_images = []
        person_files = []
        
        for img_id in range(image_range[0], image_range[1] + 1):
            filename = pattern_format.format(person_id, img_id)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                person_images.append(img)
                person_files.append(filename)
                if show_info:
                    print(f"Loaded image: {filename} (Person {person_id})")
            else:
                if show_info:
                    print(f"Warning: Could not load {filename}")
        
        if person_images:
            images.extend(person_images)
            filenames.extend(person_files)
            person_labels.extend([person_id] * len(person_images))
    
    if not images:
        raise ValueError("No images were loaded")
    
    return images, filenames, person_labels

def vectorize_images(images):
    """Convert images to column vectors."""
    height, width = images[0].shape
    vectors = []
    
    for img in images:
        # Flatten image to a column vector
        vector = img.reshape(-1)  # Equivalent to reshape(height*width)
        vectors.append(vector)
    
    # Stack vectors into a matrix (each column is an image vector)
    Wp = np.column_stack(vectors)
    return Wp, height, width

def pca_eigenspace(Wp):
    """
    Compute the PCA eigenspace following the lecture procedure.
    
    Steps from the lecture:
    1) Convert to grayscale and vectorize - Done in vectorize_images
    2) Create Wp matrix - Done in vectorize_images
    3) Calculate average vector wp
    4) Create W matrix by subtracting wp from each column of Wp
    5) Create covariance matrix C = W^T * W
    6) Compute eigenvalues and eigenvectors of C
    7) Create Ep matrix by sorting eigenvectors by eigenvalue
    8) Create eigenspace E = W * Ep
    9) Project known vectors into eigenspace PI = E^T * W
    """
    # 3) Calculate average vector wp (mean face)
    wp = np.mean(Wp, axis=1, keepdims=True)
    
    # 4) Create W matrix by subtracting wp from each column of Wp
    W = Wp - wp
    
    # 5) Create covariance matrix C = W^T * W
    C = W.T @ W
    
    # 6) Compute eigenvalues and eigenvectors of C
    eigenvalues, Ep_temp = np.linalg.eig(C)
    
    # Ensure eigenvalues are real (they might have tiny imaginary parts due to numerical precision)
    eigenvalues = np.real(eigenvalues)
    Ep_temp = np.real(Ep_temp)
    
    # 7) Create Ep matrix by sorting eigenvectors by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    Ep = Ep_temp[:, idx]
    
    # 8) Create eigenspace E = W * Ep
    E = W @ Ep
    
    # Normalize eigenvectors
    for i in range(E.shape[1]):
        norm = np.linalg.norm(E[:, i])
        if norm > 0:
            E[:, i] = E[:, i] / norm
    
    # 9) Project known vectors into eigenspace PI = E^T * W
    PI = E.T @ W
    
    return wp, W, E, PI, eigenvalues

def identify_unknown_image(unknown_path, wp, E, PI, known_images, person_labels, filenames, height, width):
    """
    Identify an unknown image using PCA.
    
    Steps from the lecture:
    1) Convert unknown image to grayscale and vectorize
    2) Subtract wp: wu = wpu - wp
    3) Project into eigenspace: PT = E^T * wu
    4) Compare PT with each vector in PI using Euclidean distance
    """
    # 1) Load unknown image and vectorize
    unknown_img = cv2.imread(unknown_path, cv2.IMREAD_GRAYSCALE)
    if unknown_img is None:
        raise ValueError(f"Could not load unknown image: {unknown_path}")
    
    wpu = unknown_img.reshape(-1, 1)  # Vectorize unknown image
    
    # 2) Subtract mean face: wu = wpu - wp
    wu = wpu - wp
    
    # 3) Project into eigenspace: PT = E^T * wu
    PT = E.T @ wu
    
    # 4) Compare PT with known projections using Euclidean distance
    min_distance = float('inf')
    min_index = -1
    distances = []
    
    for i in range(PI.shape[1]):
        # Calculate Euclidean distance between PT and PI[:, i]
        distance = np.linalg.norm(PT - PI[:, i])
        distances.append(distance)
        
        if distance < min_distance:
            min_distance = distance
            min_index = i
    
    # Identify the person
    identified_person = person_labels[min_index]
    
    # Reconstruct the identified image from eigenspace
    reconstructed_vector = wp + E @ PT
    reconstructed_img = reconstructed_vector.reshape(height, width)
    
    return min_index, identified_person, min_distance, distances, unknown_img, reconstructed_img

def display_eigenfaces(E, height, width, num_faces=4):
    """Display the top eigenfaces (principal components)."""
    plt.figure(figsize=(12, 3))
    for i in range(min(num_faces, E.shape[1])):
        # Reshape eigenface vector to image dimensions
        eigenface = E[:, i].reshape(height, width)
        
        # Normalize for display
        eigenface = eigenface - np.min(eigenface)
        eigenface = (eigenface / np.max(eigenface) * 255).astype(np.uint8)
        
        plt.subplot(1, num_faces, i+1)
        plt.imshow(eigenface)
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('eigenfaces.png')

def plot_results(unknown_img, identified_img, reconstructed_img, filenames, person_labels, min_index, identified_person, distances):
    """Display the results of the identification."""
    plt.figure(figsize=(15, 5))
    
    # Plot unknown image
    plt.subplot(1, 3, 1)
    plt.imshow(unknown_img)
    plt.title("Unknown Image")
    plt.axis('off')
    
    # Plot identified image
    plt.subplot(1, 3, 2)
    plt.imshow(identified_img)
    plt.title(f"Identified as: {filenames[min_index]}\nPerson {identified_person}")
    plt.axis('off')
    
    # Plot reconstructed image
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_img.astype(np.uint8))
    plt.title("Reconstructed Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('pca_identification_result.png')
    
    # Plot distances for all known images
    plt.figure(figsize=(12, 6))
    
    # Create a color map for different persons
    unique_persons = sorted(set(person_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_persons)))
    color_map = {person: colors[i] for i, person in enumerate(unique_persons)}
    
    # Create bars with colors based on person
    bar_colors = [color_map[person_labels[i]] for i in range(len(filenames))]
    
    bars = plt.bar(filenames, distances, color=bar_colors)
    plt.title("Euclidean Distances to Known Images")
    plt.xlabel("Image")
    plt.ylabel("Distance")
    plt.xticks(rotation=45)
    
    # Add legend for persons
    legend_patches = [plt.Rectangle((0,0),1,1, color=color_map[p]) for p in unique_persons]
    plt.legend(legend_patches, [f"Person {p}" for p in unique_persons], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('distances.png')
    
    plt.show()

def main():
    try:
        # Load known images with person labels
        known_images, filenames, person_labels = load_images()
        
        # Convert images to vectors and create Wp matrix
        Wp, height, width = vectorize_images(known_images)
        print(f"Image dimensions: {height}x{width}")
        print(f"Wp matrix shape: {Wp.shape}")
        
        # Perform PCA to get eigenspace
        wp, W, E, PI, eigenvalues = pca_eigenspace(Wp)
        print(f"Mean vector shape: {wp.shape}")
        print(f"W matrix shape: {W.shape}")
        print(f"Eigenspace E shape: {E.shape}")
        print(f"Projection PI shape: {PI.shape}")
        
        # Show eigenvalues (information content)
        total_variance = np.sum(eigenvalues)
        explained_variance = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(eigenvalues)+1), explained_variance)
        plt.title("Explained Variance Ratio")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Ratio")
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(eigenvalues)+1), cumulative_variance, '-o')
        plt.axhline(y=0.9, color='r', linestyle='--')
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance Ratio")
        plt.tight_layout()
        plt.savefig('variance_analysis.png')
        
        # Display eigenfaces (principal components)
        display_eigenfaces(E, height, width)
        
        # Display mean face
        plt.figure(figsize=(4, 4))
        mean_face = wp.reshape(height, width)
        plt.imshow(mean_face)
        plt.title("Mean Face")
        plt.axis('off')
        plt.savefig('mean_face.png')
        
        # Load and identify unknown image
        unknown_path = "unknown.bmp"
        min_index, identified_person, min_distance, distances, unknown_img, reconstructed_img = identify_unknown_image(
            unknown_path, wp, E, PI, known_images, person_labels, filenames, height, width
        )
        
        print(f"Unknown image identified as: {filenames[min_index]}")
        print(f"Identified person: {identified_person}")
        print(f"Euclidean distance: {min_distance:.4f}")
        
        # Show results
        plot_results(unknown_img, known_images[min_index], reconstructed_img, 
                    filenames, person_labels, min_index, identified_person, distances)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()