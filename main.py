import os

class Utility:
  def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

class EdgeDetection():
  def start():
    print('Edge Detection')

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
  input()