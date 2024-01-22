import cv2
import numpy as np

cap = cv2.VideoCapture('sf3.mp4')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
while True:
    ret,frame=cap.read()
    if not ret:
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       continue
    frame=cv2.resize(frame,(800,600))

    cv2.imshow("RGB",frame)
    if cv2.waitKey(100) & 0xFF == 27:
        break
cap.release()    
cv2.destroyAllWindows()     