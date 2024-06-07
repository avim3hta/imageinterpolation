import math
import csv
import matplotlib.pyplot as plt
import numpy as np

def sinc(x):
    if abs(x) < 1e-7:
        return 1.0
    else:
        return math.sin(math.pi * x) / (math.pi * x)

def lanczos(x, n):
    if x < 0:
        x = -x
    if x < n:
        return sinc(x) * sinc(x / n)
    else:
        return 0.0

def generate_lanczos_lut(n, res):
    if n < 1 or n > 8:
        raise ValueError("Unsupported Lanczos LUT order. Must be 1 <= n <= 8.")
    
    lut = [[0.0 for _ in range(res)] for _ in range(2 * n)]
    for j in range(-n, n):
        for i in range(res):
            x = j + i / res
            lut[j + n][i] = lanczos(x, n)
    
    return lut


res = 1024*8


lanczos_lut_4 = generate_lanczos_lut(4, res)

filename = "lanczosmath/lanczos_lut_4_65536.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in lanczos_lut_4:
        writer.writerow(row)


x_values = np.linspace(-4, 4, res * 8)
y_values = []

for j in range(-4, 4):
    for i in range(res):
        y_values.append(lanczos_lut_4[j + 4][i])

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Lanczos4')
plt.title('Lanczos4 Interpolation Function')
plt.xlabel('x')
plt.ylabel('Lanczos4(x)')
plt.legend()
plt.grid(True)
plt.show()
