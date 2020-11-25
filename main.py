import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# -----------------------------------------------------------------------------
# Configs
window_name = "mandelbrot"
WIDTH,HEIGHT = 800, 800
INFINITO = 0.002
ITER = 200
FAC = 0.0015
# -----------------------------------------------------------------------------

input_img = np.zeros((WIDTH,HEIGHT,3))

@jit(nopython=True)
def secMandel(c):
    z0 = complex(0,0)
    for _ in range(ITER):
        zn = z0*z0 + c
        z0 = zn

    if abs(z0) > INFINITO:
        return np.array([0,0,0])
    if abs(z0) > INFINITO/20:
        return np.array([0,255,0])
    else:
        return np.array([255,255,255])

@jit(nopython=True)
def Mandel(img,fac,dx,dy):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            img[y][x] = secMandel(complex(fac * (x-dx), fac * (y-dy)))

Mandel(input_img,FAC,WIDTH/2+200,HEIGHT/2)

plt.title(window_name+" - "+str(input_img.shape))
plt.imshow(input_img)
plt.show()