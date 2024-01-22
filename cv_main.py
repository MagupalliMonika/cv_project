import cv2
import pickle
import numpy as np
import cvzone


cap=cv2.VideoCapture('sf3.mp4')
with open("shelf","rb") as f:
        points=pickle.load(f) 
w,h=66,189
def crop(f):
     counter=0
     for i,pts in enumerate(points):
          x,y=pts
          crop=f[y:y+h,x:x+w]
          #cv2.imshow(str(x*y),crop)
          #count pixels inside each rectangle
          count=cv2.countNonZero(crop)
          cv2.putText(frame,str(count),(x,y),4,cv2.FONT_HERSHEY_PLAIN,(255,0,255),1)
          if count >700:
               cv2.rectangle(frame,pts,(pts[0]+w,pts[1]+h),(0,255,0),2)#green
          else:
               cv2.rectangle(frame,pts,(pts[0]+w,pts[1]+h),(0,0,255),2)#red
               counter=+1
          cvzone.putTextRect(frame,f'SpaceCount:-{counter}',(50,60),2,2)
    
               
    
while True:
      ret,frame=cap.read()
      if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
      frame=cv2.resize(frame,(1020,800))
      gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      frameblur=cv2.GaussianBlur(gray,(5,5),1)
      thresh=cv2.adaptiveThreshold(frameblur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,105,9)
      median=cv2.medianBlur(thresh,5)
      kernel=np.ones((3,3),np.uint8)
      dilate=cv2.dilate(median,kernel,iterations=1)
      crop(dilate)
      cv2.imshow("FRAME",frame)
      
      if cv2.waitKey(0) & 0xFF == 27:
         break
cap.release()    
cv2.destroyAllWindows()     
