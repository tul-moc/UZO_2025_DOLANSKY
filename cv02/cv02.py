import numpy as np
import cv2
import os

def track_object(frame, roi_hist, window):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)

    x, y, w, h = window

    while True:
        mask = np.zeros_like(back_proj)
        mask[y:y + h, x:x + w] = back_proj[y:y + h, x:x + w]

        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            new_x = cx - w // 2
            new_y = cy - h // 2

            new_x = max(0, min(new_x, frame.shape[1] - w))
            new_y = max(0, min(new_y, frame.shape[0] - h))
        else:
            break

        if new_x == x and new_y == y:
            break

        x, y = new_x, new_y

    return (x, y, w, h)

if __name__ == '__main__':
    clear_screen = lambda: os.system('cls' if os.name == 'nt' else 'clear')
    clear_screen()

    video_capture = cv2.VideoCapture('cv02_hrnecek.mp4')
    template_image = cv2.imread("cv02_vzor_hrnecek.bmp")
    template_hsv = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)

    roi_hist = cv2.calcHist([template_hsv], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    frame_height, frame_width = video_capture.read()[1].shape[:2]
    initial_x, initial_y, initial_w, initial_h = frame_width // 2 - 50, frame_height // 2 - 50, 100, 150
    tracking_window = (initial_x, initial_y, initial_w, initial_h)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or error loading frame.")
            break

        tracking_window = track_object(frame, roi_hist, tracking_window)

        x, y, w, h = tracking_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()
