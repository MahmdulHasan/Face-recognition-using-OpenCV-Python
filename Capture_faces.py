import cv2
import numpy as np
detector=cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
cam = cv2.VideoCapture(0)
Id=raw_input('enter your id: ')
sampleNum=0
ret, img = cam.read()
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',gray)
        sampleNum=sampleNum+1
        frame=gray[y:y+h,x:x+w]
        
        cv2.imwrite("images/User."+Id +'.'+ str(sampleNum) + ".jpg", frame)

        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
