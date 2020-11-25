import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Configs
window_name = "mandelbrot"
WIDTH,HEIGHT = 800, 800
INFINITO = 1000
ITER = 2000
FAC = 0.001
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

for x in range(WIDTH):
    for y in range(HEIGHT):
        input_img[x][y] = secMandel(complex(FAC * x, FAC * y))



plt.title(window_name+" - "+str(input_img.shape))
plt.imshow(input_img,cmap='gray',clim=(0,255))
plt.show()