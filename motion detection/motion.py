import cv2,time

video = cv2.VideoCapture(0)ض
first_frame=None
count=-1
while True:
    count = count + 1
    print(count)
    check,frame = video.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(gray,(21,21),0)
    if count%100 == 0 :
        first_frame=gray

    delta_frame= cv2.absdiff(first_frame,gray)
    threshold_frame = cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame,None,iterations=2)

    (cntr,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for countour in cntr:
        if cv2.contourArea(countour)<1000:
            continue
        (x,y,w,h)= cv2.boundingRect(countour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        print("woow there is a movement ")



    cv2.imshow("cvghj",frame)
    cv2.imshow('Threshold(foreground mask)', delta_frame)
    cv2.imshow('Frame_delta', threshold_frame)
    key=cv2.waitKey(1)
    if  key == ord('q'):
        break


video.release()
cv2.destroyWindow()