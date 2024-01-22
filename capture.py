import time
import cv2
cpt=0
maxFrames=50
count=0
cap=cv2.VideoCapture('sf3.mp4')
while cpt<maxFrames:
    ret,frame=cap.read()
    if not ret:
        break
    count+=1
    if count%1!=0:
        continue
    frame=cv2.resize(frame,(1080,500))
    cv2.imshow("f",frame)
    cv2.imwrite(r"images/img-b_%d.jpg" %cpt,frame)
    cpt+=1
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()