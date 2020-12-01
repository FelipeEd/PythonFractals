from mandelbrot import *


# Iniciando coisas do pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption(window_name)

mandel = Mandelbrot()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()


    mandel.move()
    mandel.render()

    surface = pygame.surfarray.make_surface(mandel.getImg())

    screen.blit(surface, (10, 0))
    pygame.display.flip()
    clock.tick(60)

