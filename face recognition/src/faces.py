import numpy as np
import cv2
import pickle





face_cascade  =  cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

face_recognizer =  cv2.face.LBPHFaceRecognizer_create() 
face_recognizer.read("trainner.yml")
lables = {"person_name": 1}

with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        lables    = {v:k for k,v in og_labels.items()}

                        
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #hadi bah nerodoh gray 
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        id_ , conf = face_recognizer.predict(roi_gray)
        if conf >= 50 :# and conf <= 85 :
            print(id_)
            print(lables[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name =  lables[id_]
            color = (255 ,255,255)
            strock =  2 
            cv2.putText(frame, name , (x,y), font ,1 , color, strock , cv2.LINE_AA )
            
        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)
        color =  (255,0,0)#BGR 0-255
        Strock = 2  
        end_cord_x   =  x + y
        end_cord_y =  y + h
        cv2.rectangle(frame, (x,y), (end_cord_x , end_cord_y), color , Strock)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes : 
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0) , 2)
            
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
