import cv2
import numpy as np
import math
import threading
import pickle
import os


res = 128
fixed_width = 1280
fixed_height = 960

def Sinc(x):
    x *= math.pi
    return math.sin(x) / x if abs(x) > 1.0e-07 else 1.0

def Lanczos(x, n):
    if x < 0:
        x = -x
    if x < n:
        return Sinc(x) * Sinc(x / n)
    return 0

mutex = threading.Lock()

def InitializeLanczosIntLUT(n):
    global int_lut
    lut_filename = 'lanczos_lut_n4.pkl'
    
    with mutex:
        if not hasattr(InitializeLanczosIntLUT, 'int_lut'):
            InitializeLanczosIntLUT.int_lut = {}

        if n in InitializeLanczosIntLUT.int_lut:
            return InitializeLanczosIntLUT.int_lut[n]

        if os.path.exists(lut_filename):
            with open(lut_filename, 'rb') as f:
                LUT = pickle.load(f)
        else:
            LUT = [0] * (n * res + 1)
            for i in range(n):
                for j in range(res):
                    k = i * res + j
                    LUT[k] = Lanczos(i + j / res, n)
            LUT[n * res] = 0

            with open(lut_filename, 'wb') as f:
                pickle.dump(LUT, f)

        InitializeLanczosIntLUT.int_lut[n] = LUT

    return LUT

def LanczosInterpolation(x, lut, n):
    if x < 0:
        x = -x
    if x < n:
        index = int(x * res)
        return lut[index]
    return 0

def precompute_lanczos_weights(fixed_width, fixed_height, zoom_factor, lut, n):
    x_coords = (np.arange(fixed_width) + 0.5) / zoom_factor - 0.5
    y_coords = (np.arange(fixed_height) + 0.5) / zoom_factor - 0.5
    int_x = np.floor(x_coords).astype(int)
    int_y = np.floor(y_coords).astype(int)
    frac_x = x_coords - int_x
    frac_y = y_coords - int_y

    weights_x = np.zeros((fixed_width, 7))
    weights_y = np.zeros((fixed_height, 7))

    for x in range(fixed_width):
        for i in range(-3, 4):
            weights_x[x, i + 3] = LanczosInterpolation(i - frac_x[x], lut, n)
    
    for y in range(fixed_height):
        for j in range(-3, 4):
            weights_y[y, j + 3] = LanczosInterpolation(j - frac_y[y], lut, n)

    return int_x, int_y, weights_x, weights_y

def lanczos_digital_zoom(image, zoom_factor):
    height = fixed_height
    width = fixed_width
    center_width = int(width / zoom_factor)
    center_height = int(height / zoom_factor)
    start_x = (width - center_width) // 2
    start_y = (height - center_height) // 2
    end_x = start_x + center_width
    end_y = start_y + center_height
    cropped_image = image[start_y:end_y, start_x:end_x]
    lut = InitializeLanczosIntLUT(4)
    int_x, int_y, weights_x, weights_y = precompute_lanczos_weights(fixed_width, fixed_height, zoom_factor, lut, 4)
    zoomed_image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            result = np.zeros(3)
            for i in range(-3, 4):
                for j in range(-3, 4):
                    px = int_x[x] + i
                    py = int_y[y] + j
                    if 0 <= px < cropped_image.shape[1] and 0 <= py < cropped_image.shape[0]:
                        lanczos_weight = weights_x[x, i + 3] * weights_y[y, j + 3]
                        result += lanczos_weight * cropped_image[py, px]
            zoomed_image[y, x] = np.clip(result, 0, 255)

    return zoomed_image

if __name__ == "__main__":
    video_path = "img/output_1280x960.avi"  # Change this to the path of your video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('zoomed_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (fixed_width, fixed_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        zoom_factor = 8
        zoomed_frame = lanczos_digital_zoom(frame, zoom_factor)
        out.write(zoomed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
