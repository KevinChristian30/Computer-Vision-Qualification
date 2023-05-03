import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Utility:
  def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

class EdgeDetection:
  def showResult(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))

    for i, (label, image) in enumerate(res_stack):
      plt.subplot(nrow, ncol, i + 1)
      plt.imshow(image, cmap='gray')
      plt.title(label)
      plt.axis('off')
  
    plt.show()

  def start():
    image = cv2.imread('./Edge Detection/rickroll.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplace8u = cv2.Laplacian(gray_image, cv2.CV_8U)
    laplace16s = cv2.Laplacian(gray_image, cv2.CV_16S)
    laplace32f = cv2.Laplacian(gray_image, cv2.CV_32F)
    laplace64f = cv2.Laplacian(gray_image, cv2.CV_64F)

    laplace_labels = ['8U', '16S', '32F', '64F']
    laplace_images = [laplace8u, laplace16s, laplace32f, laplace64f]

    EdgeDetection.showResult(2, 2, zip(laplace_labels, laplace_images))

class ShapeDetection:
  def start():
    print('Shape Detection')

class PatternDetection:
  def start():
    print('Pattern Recognition')

class FaceDetection:
  def start():
    print('Face Detection')

class MenuFacade:
  def displayMenu():
    print('Computer Vision Qualification')
    print('=============================')
    print('1. Edge Detection') 
    print('2. Shape Detection') 
    print('3. Pattern Recognition') 
    print('4. Face Detection') 
    print('5. Exit')

  def exitScreen():
    Utility.clear()
    input('Thank You! Press Enter to Exit')
    exit(0)

  def route(input):
    if (input == '1'): 
      EdgeDetection.start()
    elif (input == '2'): 
      ShapeDetection.start()
    elif (input == '3'): 
      PatternDetection.start()
    elif (input == '4'): 
      FaceDetection.start()
    elif (input == '5'):
      MenuFacade.exitScreen()

while (True):
  Utility.clear()
  MenuFacade.displayMenu()
  option = input('>> ')
  MenuFacade.route(option)