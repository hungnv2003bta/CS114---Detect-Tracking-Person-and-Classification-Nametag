import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from datetime import datetime
import os
from FINAL_classification_predict import ImagePredictor

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

def imgwrite(img):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = '%s.jpg' % current_time
    predictor = ImagePredictor()
    result_predict = predictor.predict_image(img)

    if result_predict == "with_name_tag":
        cv2.imwrite(os.path.join("results_with_name_tag", filename), img)
    else:
        cv2.imwrite(os.path.join("results_without_name_tag", filename), img)

        
def main():
    model = YOLO('/Users/hungnguyen/Projects/python/yolov8/model/yolov8m.pt')

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    # SOURCE
    cap=cv2.VideoCapture('test_vid/IMG_2464.mp4')

    class_list = ['person']
    count=0
    tracker=Tracker()   
    #change area here, A,D,C,B (x,y)
    area=[(0,820),(0,845),(720,735),(720,710)] #IMG_2464.mp4
    area_c=set()

    while True:    
        ret,frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        
        frame=cv2.resize(frame,(720,1280)) #test_vid/IMG_2445.mp4

        results=model.predict(frame, conf=0.5, classes=0)
    #   print(results)
        a=results[0].boxes.data
        px=pd.DataFrame(a).astype("float")
    #    print(px)
        list=[]
        for index,row in px.iterrows():
            print(row)

            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_list[d]
            if 'person' in c:
                list.append([x1,y1,x2,y2])
        
        bbox_idx=tracker.update(list)
        for bbox in bbox_idx:
            x3,y3,x4,y4,id=bbox
            results=cv2.pointPolygonTest(np.array(area,np.int32),((x4,y4)),False)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
            cv2.putText(frame,f'id:{id}',(x3,y3),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 255),2)
            if results>=0 and id not in area_c:
                crop=frame[y3:y4,x3:x4]
                imgwrite(crop)
                # cv2.imshow(str(id),crop) 
                area_c.add(id)
        cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
        print(area_c)
        k=len(area_c)
        cv2.putText(frame,str(k),(50,60),cv2.FONT_HERSHEY_PLAIN,5,(255,0,255),3)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1)&0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
