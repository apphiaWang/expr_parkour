import pygame.camera
import pygame.image
import sys

# init camera
pygame.camera.init()
cameras = pygame.camera.list_cameras()
print("Using camera %s ..." % cameras[0])
webcam = pygame.camera.Camera(cameras[0])
webcam.set_controls(hflip = True, vflip = False)
webcam.start()

# init pygame
pygame.init()
gameDisplay = pygame.display.set_mode((640,480))

while True:
   # display webcam captured images in pygame window
   img = webcam.get_image()
   gameDisplay.blit(img,(0,0))
   pygame.display.update()

   # exit
   for event in pygame.event.get() :
      if event.type == pygame.QUIT :
         webcam.stop()
         pygame.quit()
         exit()
