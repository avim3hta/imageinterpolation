import cv2
import numpy as np 
#video="path"
cap = cv2.VideoCapture(0) #videofeed set to camera
def zoom_center(frame, zoom_factor=8):
    y_size = frame.shape[0]
    x_size = frame.shape[1]
        
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    img_cropped = frame[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor,interpolation=cv2.INTER_LANCZOS4) #change interpolation algorithm used cv2._____


while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    image = zoom_center(frame)
    cv2.imshow('Frame',image) 
    cv2.imshow('live',frame) #both frames for comparison
    if cv2.waitKey(25) & 0xFF == ord('q'): #q to break camera feed
      break
  else: 
    break
cap.release()
cv2.destroyAllWindows()


    