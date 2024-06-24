import csv
import threading
import os
mutex = threading.Lock()
res = 1028
def InitializeLanczosIntLUT(n):
    global int_lut

    lut_filename = 'lanczos_lut_n4_big.csv'
    
    with mutex:
        if not hasattr(InitializeLanczosIntLUT, 'int_lut'):
            InitializeLanczosIntLUT.int_lut = {}

        if n in InitializeLanczosIntLUT.int_lut:
            return InitializeLanczosIntLUT.int_lut[n]

        if os.path.exists(lut_filename):
            with open(lut_filename, 'r') as f:
                reader = csv.reader(f)
                LUT = []
                for row in reader:
                    LUT.append([float(x) for x in row])
        else:
            LUT = [0] * (n * res + 1)
            for i in range(n):
                for j in range(res):
                    k = i * res + j
                    LUT[k] = Lanczos(i + j / res, n)
            LUT[n * res] = 0

            with open(lut_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(LUT)

        InitializeLanczosIntLUT.int_lut[n] = LUT

    return LUT
