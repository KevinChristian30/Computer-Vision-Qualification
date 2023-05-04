import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import *

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
  sides_to_shape = {
    '3': 'Triangle',
    '4': 'Rectangle',
    '5': 'Pentagon',
    '6': 'Hexagon',
    '7': 'Heptagon',
    '8': 'Octagon',
    '9': 'Nonagon',
    '10': 'Decagon'
  }

  def start():
    image = cv2.imread('./Shape Detection/shapes.png')
    new_image = cv2.resize(image, (800, 400))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    first = False
    for contour in contours:
      if first == False:
        first = True
        continue

      approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
      M = cv2.moments(contour)
      x = int(M['m10'] / M['m00'])
      y = int(M['m01'] / M['m00'])
      
      if (len(approx) >= 3 and len(approx) <= 10):
        cv2.putText(new_image, ShapeDetection.sides_to_shape[str(len(approx))], (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
      else:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.8:
          cv2.putText(new_image, 'Circle', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))

    cv2.imshow('Shape Detection', new_image)
    cv2.waitKey()

class PatternDetection:
  def start():
    classifier = cv2.CascadeClassifier('./Pattern Recognition/haarcascade_frontalface_default.xml')

    train_path = './Pattern Recognition/train'
    tdir = os.listdir(train_path)

    face_list = []
    class_list = []

    for idx, train_dir in enumerate(tdir):
      for image_path in os.listdir(f'{train_path}/{train_dir}'):
        path = f'{train_path}/{train_dir}/{image_path}'
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        faces = classifier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

        if len(faces) < 1:
          continue
        else:
          for face_rect in faces:
            x, y, w, h = face_rect
            face_image = gray[y:y+w, x:x+h]
            face_list.append(face_image)
            class_list.append(idx)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))

    test_path = './Pattern Recognition/test'

    for path in os.listdir(test_path):
      fullpath = f'{test_path}/{path}'
      image = cv2.imread(fullpath)
      igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      faces = classifier.detectMultiScale(igray, scaleFactor = 1.2, minNeighbors = 5)

      if len(faces) < 1:
        continue
      
      for face_rect in faces:
        x, y, w, h = face_rect
        face_image = igray[y:y+w, x:x+h]
        res, conf = face_recognizer.predict(face_image)

        conf = math.floor(conf * 100) / 100
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        image_text = f'{tdir[res]} : {str(conf)}%'

        cv2.putText(image, image_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1)
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

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