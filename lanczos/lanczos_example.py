import numpy as np


a = 4 
lookup_size = 1000  
x_max = a  


x_values = np.linspace(-x_max, x_max, lookup_size)
sinc_values = np.sinc(x_values)  


def lanczos_interpolation(x, lookup_table, x_vals, a):
    if abs(x) >= a:
        return 0
    index = int((x + x_max) / (2 * x_max) * (lookup_size - 1))
    sinc_x = lookup_table[index]
    sinc_x_a = lookup_table[int((x / a + x_max) / (2 * x_max) * (lookup_size - 1))]
    return sinc_x * sinc_x_a


x = 2.5  
interpolated_value = lanczos_interpolation(x, sinc_values, x_values, a)
print(f"Lanczos interpolation at x = {x}: {interpolated_value}")


def apply_lanczos_interpolation(data, factor, a=4):
    out_size = int(len(data) * factor)
    output = np.zeros(out_size)

 
    scale = len(data) / out_size

    for i in range(out_size):
        x = i * scale
        left = int(np.floor(x)) - a + 1
        right = int(np.floor(x)) + a
        sum_lanczos = 0
        sum_weights = 0

        for j in range(left, right + 1):
            if 0 <= j < len(data):
                weight = lanczos_interpolation(x - j, sinc_values, x_values, a)
                sum_lanczos += data[j] * weight
                sum_weights += weight

        if sum_weights != 0:
            output[i] = sum_lanczos / sum_weights

    return output


data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
zoom_factor = 2 
zoomed_data = apply_lanczos_interpolation(data, zoom_factor)
print(f"Zoomed data: {zoomed_data}")
