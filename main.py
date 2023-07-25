import numpy as np
import cv2

cascade_src = 'cars.xml'
video = 'data/Cars_On_Highway.mp4'
# video = 'data/video1.avi'
# video = 'data/video2.avi'


def detectCars(filename):
  rectangles = []
  cascade = cv2.CascadeClassifier(cascade_src)

  vc = cv2.VideoCapture(filename)

  if vc.isOpened():
      rval , frame = vc.read()
  else:
      rval = False

  car_count = 0  # Initialize car count variable
  detection_timer = 0  # Timer to delay incrementing car count


  while rval:
    rval, frame = vc.read()
    frameHeight, frameWidth, fdepth = frame.shape

    # Resize
    frame = cv2.resize(frame, ( 600,  400 ))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # haar detection.
    cars = cascade.detectMultiScale(gray, 1.3, 3)


    for (x, y, w, h) in cars:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
       # Increment the car count only when a new detection occurs
      if detection_timer == 0:
        car_count += 1
        detection_timer = 20  # Set a delay of 20 frames (adjust as needed)

      cv2.putText(frame, f'Car Count: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
       # Decrement the detection timer to control the counting delay
      if detection_timer > 0:
            detection_timer -= 1


    # show result
    cv2.imshow("Result",frame)

    if cv2.waitKey(33) == ord('q'):
      break

  vc.release()
  cv2.destroyAllWindows()



detectCars(video)

