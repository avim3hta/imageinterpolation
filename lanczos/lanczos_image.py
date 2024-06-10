import cv2
import numpy as np
import math
import threading
import pickle
import os


res = 128

# Lanczos functions
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
    lut_filename = 'lanczos_lut_n4_round.pkl'
    
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


def lanczos_digital_zoom(image, zoom_factor):
    height, width, _ = image.shape
    center_width = int(width / zoom_factor)
    center_height = int(height / zoom_factor)
    start_x = (width - center_width) // 2
    start_y = (height - center_height) // 2
    end_x = start_x + center_width
    end_y = start_y + center_height
    cropped_image = image[start_y:end_y, start_x:end_x]
    lut = InitializeLanczosIntLUT(4)  
    zoomed_image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            orig_x = (x + 0.5) / zoom_factor - 0.5
            orig_y = (y + 0.5) / zoom_factor - 0.5
            int_x = int(np.floor(orig_x))
            int_y = int(np.floor(orig_y))
            frac_x = orig_x - int_x
            frac_y = orig_y - int_y
            result = np.zeros(3)
            for i in range(-3, 4):
                for j in range(-3, 4):
                    px = int_x + i
                    py = int_y + j
                    if 0 <= px < cropped_image.shape[1] and 0 <= py < cropped_image.shape[0]:
                        lanczos_weight = LanczosInterpolation(i - frac_x, lut, 4) * LanczosInterpolation(j - frac_y, lut, 4)
                        result += lanczos_weight * cropped_image[py, px]
            zoomed_image[y, x] = np.clip(result, 0, 255)
    return zoomed_image

if __name__ == "__main__":
    original_image = cv2.imread("img/image.jpg")
    zoom_factor = 8 
    zoomed_image = lanczos_digital_zoom(original_image, zoom_factor)
    cv2.imwrite("c.jpg", zoomed_image)
