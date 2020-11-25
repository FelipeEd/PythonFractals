import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Configs
window_name = "mandelbrot"
WIDTH,HEIGHT = 800, 800
INFINITO = 100
ITER = 2000

# -----------------------------------------------------------------------------

input_img = np.zeros((WIDTH,HEIGHT))

def secMandel(c):
    z0 = 0
    for _ in range(ITER):
        zn = z0*z0 + c
        z0 = zn
    if abs(z0) > INFINITO:
        return 0
    else:
        return 255

for pixel in input_img:
    print(pixel[0],pixel[1])
    pixel = secMandel(complex(pixel[0],pixel[1]))



plt.title(window_name+" - "+str(input_img.shape))
plt.imshow(input_img, clim=(0,255))
plt.show()