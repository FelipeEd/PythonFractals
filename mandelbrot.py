from config import *
import numpy as np
from numba import jit
import pygame

# Funções fora da classe para usar o numba já que por algum motivo
# o numba não consegue trasformar a classe Mandelbrot em codigo C
@jit(nopython=True)
def secMandel(c):
    z0 = complex(0, 0)
    for _ in range(ITER):
        zn = z0 * z0 + c
        z0 = zn
        if abs(z0) > INFINITO:
            for n in range(200):
                if _ < n:
                    return np.array([(n * 10) % 255, (n * 20) % 255, (n * 30) % 255], dtype=np.uint8)
            return np.array([100, 100, 100], dtype=np.uint8)
    else:
        return np.array([0, 0, 0], dtype=np.uint8)


@jit(nopython=True)
def calcSet(img,fac,dx,dy):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            img[x][y] = secMandel(complex(fac * (x - dx), fac * (y - dy)))
    return img


class Mandelbrot:


    def __init__(self):
        self.title = window_name
        self.width = WIDTH
        self.height = HEIGHT
        self.img = np.ones((WIDTH,HEIGHT,3),dtype=np.uint8)
        self.iter = ITER
        self.att = True
        # Zoom
        self.fac = FAC

        # Translação
        self.dx = WIDTH / 2 + 200
        self.dy = HEIGHT / 2

        # Speed
        self.tSpeed = 1
        self.sSpeed = 1

    def getImg(self):
        return self.img


    def render(self):

        self.img = calcSet(self.img,self.fac,self.dx,self.dy)
        self.att = False

    def move(self):
        keys = pygame.key.get_pressed()

        # Controle do jogo

        if keys[pygame.K_w]:
            self.dy += 10 * self.tSpeed
        if keys[pygame.K_a]:
            self.dx += 10 * self.tSpeed
        if keys[pygame.K_s]:
            self.dy -= 10 * self.tSpeed
        if keys[pygame.K_d]:
            self.dx -= 10 * self.tSpeed
        if keys[pygame.K_LEFT]:
            self.tSpeed +=3
        if keys[pygame.K_RIGHT]:
            self.tSpeed -= 3
            if self.tSpeed < 0:
                self.tSpeed = 0

        if keys[pygame.K_SPACE]:
            self.att = True
        if keys[pygame.K_z]:
            self.fac *= 0.90
        if keys[pygame.K_x]:
            self.fac *= 1.005
