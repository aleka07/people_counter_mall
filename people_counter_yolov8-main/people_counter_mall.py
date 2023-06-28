import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*


model=YOLO('yolov8x.pt')


area1=[(380, 556),(647, 681),(605, 687),(356, 563)]

area2=[(348, 565),(595, 690),(560, 701),(312, 571)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('peoplecount1.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count = 0
people_entering = {}
people_exiting = {}

entering = set()
exiting = set()

tracker = Tracker()
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1280,720))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
 #   print(results)
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
           list.append([x1,y1,x2,y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
           x3,y3,x4,y4,id=bbox
           results = cv2.pointPolygonTest(np.array(area2,np.int32), ((x4,y4)), False)

           if results>=0:
                people_entering[id] = (x4,y4)

           if id in people_entering:
                results_1 = cv2.pointPolygonTest(np.array(area1,np.int32), ((x4,y4)), False)
                
                if results_1 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (182, 208, 52), 2)
                    cv2.circle(frame,(x4,y4),3,(0, 255, 255),-1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                    entering.add(id)

            #people exiting
           results = cv2.pointPolygonTest(np.array(area1,np.int32), ((x4,y4)), False)

           if results>=0:
                people_exiting[id] = (x4,y4)

           if id in people_exiting:
                results_3 = cv2.pointPolygonTest(np.array(area2,np.int32), ((x4,y4)), False)
                
                if results_3 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (182, 208, 52), 2)
                    cv2.circle(frame,(x4,y4),3,(0, 255, 255),-1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                    exiting.add(id)
        
      
            
            
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255, 200, 170),2)
    cv2.putText(frame, '1', (657, 688), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2)
    cv2.putText(frame, f'Entering: {len(entering)}', (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Exiting: {len(exiting)}', (60,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255, 180, 200),2)
    cv2.putText(frame, '2', (585, 703), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

