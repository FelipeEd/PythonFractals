from config import *
import numpy as np
from numba import jit
import pygame
from cv2 import imwrite,rotate,ROTATE_90_COUNTERCLOCKWISE
import math

# Funções fora da classe para usar o numba já que por algum motivo
# o numba não consegue trasformar a classe Mandelbrot em codigo C
@jit(nopython=True)
def secMandel(c,iter):
    z0 = complex(0, 0)
    for _ in range(iter):
        zn = z0 * z0 + c
        z0 = zn
        if abs(z0) > INFINITO:
            n = _
            return np.array([(n * 1) % 255, (n * 2) % 255, (n * 3) % 255], dtype=np.uint8)

    else:
        return np.array([0, 0, 0], dtype=np.uint8)

# Otimização para eliminar os elementos do cardioid
@jit(nopython=True)
def inCardioid(c):
    x = c.real
    y = c.imag
    p = np.sqrt((x-1/4)**2+y**2)
    pc = 1/2-1/2*(np.cos(np.arctan(y/(x-1./4))))
    if p <= pc:
        return True
    else:
        return False

@jit(nopython=True)
def calcSet(img,linx,liny,iter):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            c = complex(linx[x],liny[y])
            if inCardioid(c):
                img[x][y] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                img[x][y] = secMandel(c,iter)
    return img


class Mandelbrot:


    def __init__(self):
        self.title = window_name

        self.a =-1
        self.b = 1
        self.c = -1 / PROPORCAO
        self.d = 1 / PROPORCAO
        # rect [a,b]x[c,d]
        self.linx = np.linspace(self.a,self.b,WIDTH)
        self.liny = np.linspace(self.c,self.d,HEIGHT)

        self.img = np.ones((WIDTH,HEIGHT,3),dtype=np.uint8)
        self.iter = ITER

        self.fact = FACT
        self.facz = FACZ


    def getImg(self):
        return self.img


    def render(self):
        self.img = calcSet(self.img,self.linx,self.liny,ITER)

    def print(self):
        print("Printing...")
        plinx = np.linspace(self.a, self.b, RENDER[0])
        pliny = np.linspace(self.c, self.d, RENDER[1])
        pimg = np.ones((RENDER[0],RENDER[1],3),dtype=np.uint8)
        pimg = calcSet(pimg,plinx,pliny,5000)
        imwrite("Mandelbrot.png", pimg)
        print("DONE!")

    def move(self,direction):
        if direction == 'up':
            self.liny -= (self.liny[-1] - self.liny[0]) * self.fact
        if direction == 'down':
            self.liny += (self.liny[-1] - self.liny[0]) * self.fact
        if direction == 'left':
            self.linx -= (self.linx[-1] - self.linx[0]) * self.fact
        if direction == 'right':
            self.linx += (self.linx[-1] - self.linx[0]) * self.fact

        self.a = self.linx[0]
        self.b = self.linx[-1]
        self.c = self.liny[0]
        self.d = self.liny[-1]


    def zoom(self,direction):

        if direction == 'in':
            self.a += (self.b - self.a) * self.facz
            self.b -= (self.b - self.a) * self.facz
            self.c += (self.d - self.c) * self.facz
            self.d -= (self.d - self.c) * self.facz
            self.linx = np.linspace(self.a,self.b, WIDTH)
            self.liny = np.linspace(self.c, self.d, HEIGHT)

        if direction == 'out':
            self.a -= (self.b - self.a) * self.facz
            self.b += (self.b - self.a) * self.facz
            self.c -= (self.d - self.c) * self.facz
            self.d += (self.d - self.c) * self.facz
            self.linx = np.linspace(self.a, self.b, WIDTH)
            self.liny = np.linspace(self.c, self.d, HEIGHT)



    def control(self):
        keys = pygame.key.get_pressed()

        # Controle do jogo

        if keys[pygame.K_w]:
            self.move('up')
        if keys[pygame.K_a]:
            self.move('left')
        if keys[pygame.K_s]:
            self.move('down')
        if keys[pygame.K_d]:
            self.move('right')
        if keys[pygame.K_RIGHT]:
            self.fact += 0.1
            if self.fact > 0.9:
                self.fact = 0.9
        if keys[pygame.K_LEFT]:
            self.fact -= 0.1
            if self.fact < 0:
                self.fact = 0
        if keys[pygame.K_SPACE]:
            self.print()
        if keys[pygame.K_z]:
            self.zoom('out')
        if keys[pygame.K_x]:
            self.zoom('in')
