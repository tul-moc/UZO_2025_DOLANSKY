import numpy as np
import cv2
import argparse

def argparser():
    parser = argparse.ArgumentParser(description="Process an integer argument.")
    parser.add_argument('integer', type=int, help='An integer argument')
    args = parser.parse_args()
    return args.integer

def main():
    integer_arg = argparser()
    print(f"Integer argument received: {integer_arg}")
    image = cv2.imread("cv03_robot.bmp")
    
    # Get the image dimensions (height, width)
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Calculate the new bounding dimensions of the image
    angle_rad = np.deg2rad(integer_arg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    new_w = int(abs(h * sin_angle) + abs(w * cos_angle))
    new_h = int(abs(h * cos_angle) + abs(w * sin_angle))
    
    # Calculate the new center of the image
    new_center = (new_w // 2, new_h // 2)
    
    # Perform the rotation manually using a for loop
    rotated = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    
    for y in range(h):
        for x in range(w):
            # Translate coordinates to origin
            x_translated = x - center[0]
            y_translated = y - center[1]
            
            # Rotate coordinates
            x_rotated = int(x_translated * cos_angle - y_translated * sin_angle + new_center[0])
            y_rotated = int(x_translated * sin_angle + y_translated * cos_angle + new_center[1])
            
            # Check if the new coordinates are within the image bounds
            if 0 <= x_rotated < new_w and 0 <= y_rotated < new_h:
                rotated[y_rotated, x_rotated] = image[y, x]
    
    cv2.imshow("robot", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()