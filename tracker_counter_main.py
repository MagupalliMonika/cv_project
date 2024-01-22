import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
from yolo_det import YOLO_D
import cvzone
import cv2
from tracker import*
tracker=Tracker()
polylines = [
    [(59, 5), (792, 5), (792, 296), (59, 296)],
    [(59, 306), (792, 306), (792, 545), (59, 545)]
]
area_names = ["shelf1", "shelf2"]

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model = YOLO_D('results/r1/weights/best.pt')

cap = cv2.VideoCapture("sf3.mp4")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (800, 600))
    frame_copy = frame.copy()
    bboxes, classes, scores = model.detect(frame)
    bbox_idx=tracker.update(bboxes)
    #print(bbox_idx)
    shelfs=[]
    removed=[]
    counter2=[]
    counter1=[]
    shelf_items_count = {}
    for i,polyline in enumerate(polylines):
            shelfs.append(i)
            shelf_items_count[i] = {'count': 0, 'id': [],'cords':[]}
            cv2.polylines(frame, [np.array(polyline, np.int32).reshape((-1, 1, 2))], True, (0, 0, 0), 2)
            #cvzone.putTextRect(frame, f'{i}', tuple(polyline[0]), 0.2, 0)
            for bbox in bbox_idx:
                x3,y3,x4,y4,id=bbox
                #cv2.rectangle(frame,(x3,y3),(x4,y4),(0.0.255),1)
                #cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
                cx=int(x3+x4)//2
                cy=int(y3+y4)//2
                result=cv2.pointPolygonTest(np.array(polyline,np.int32).reshape((-1, 1, 2)),((cx,cy)),False)
                if result>=0:
            #       cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)    
            #           cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
                    #cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
                    #cvzone.putTextRect(frame, f'{id}', (x3,y3),0.5,0)
                    
                    counter1.append(id)
                    shelf_items_count[i]['count'] += 1
                    shelf_items_count[i]['id'].append(id)
                    shelf_items_count[i]['cords'].append([x3,y3,x4,y4,cx,cy])

                else:
                    removed.append({'cx': id, 'shelf': i})
                    counter2.append(id)
                      
    #cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
    #cv2.line(frame,(470,4),(470,484),True,(255,0,0),2)
    ca1=len(counter1)
    shels_c=len(shelfs)
    free_space = len(counter2)-len(counter1)
    cvzone.putTextRect(frame, f'total_count:-{ca1}', (70,70),1,1,(255, 255, 255), 2)
    cvzone.putTextRect(frame, f'shelfs_count:-{shels_c}', (70,30),1,1,(255, 255, 255), 2)
    #cvzone.putTextRect(frame, f'ITEM_REMOVED_COUNT: {free_space}', (20, 120), 1, 1)
    # Print item count for each shelf
    for shelf, values in shelf_items_count.items():
        shelf_no = shelf + 1
        present = values['count']
        cords = values['cords']
        id_points = ', '.join(map(str, values['id']))
        #print(f'Shelf {shelf_no} Items: {present}, ids : {id_points}')
        cvzone.putTextRect(frame, f'Shelf_{shelf_no} Items:-{present} ids:-{id_points}', (70, 110 + 30 * shelf), 1, 1,(255, 255, 255), 2)
        #cvzone.putTextRect(frame, f'CX Points: {cx_points}', (20, 180 + 30 * shelf), 1, 1)
        for cord in cords:
            x1, y1, x2, y2,cx,cy = cord
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(values['id'][cords.index(cord)]), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("RGB",frame)
    '''
    results = model.predict(frame, device="cpu")
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []
    for index, row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        c = class_list[d]
        if 'item' in c:
            list1.append([x1, y1, x2, y2, cx, cy])

    counter1 = []
    counter2 = []
    lots = []
    removed = []

    for i, polyline in enumerate(polylines):
        lots.append(i)
        cv2.polylines(frame, [np.array(polyline, np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{i}', tuple(polyline[0]), 0.5, 1)

        for i in list1:
            x1, y1, x2, y2, cx1, cy1 = i
            result = cv2.pointPolygonTest(np.array(polyline, np.int32).reshape((-1, 1, 2)), (cx1, cy1), False)
            result1 = cv2.pointPolygonTest(np.array(polyline, np.int32).reshape((-1, 1, 2)), (cx1, cy1), True)
            #print(result1,(cx1,cy1))
            if result >= 0:
                counter1.append(cx1)
            else:
                removed.append([x1, y1, x2, y2, cx1, cy1])
                counter2.append(cx1)

    for i in removed:
        x1, y1, x2, y2, cx1, cy1 = i
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    shels_count = len(lots)
    filled_count = len(counter1)
    free_space = len(counter2)
    #print(free_space,counter2)
    cvzone.putTextRect(frame, f'SHELFS_COUNT: {shels_count}', (50, 30), 1, 1)
    cvzone.putTextRect(frame, f'ITEMS_COUNT: {filled_count}', (50, 60), 1, 1)
    cvzone.putTextRect(frame, f'ITEM_REMOVED_COUNT: {free_space}', (50, 90), 1, 1)
    '''
    #cv2.imshow('FRAME', frame)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()