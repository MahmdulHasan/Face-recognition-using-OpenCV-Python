import cv2                                                          


face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')


recognise = cv2.createEigenFaceRecognizer(40,2650)  
recognise.load("trainingDataEigan.xml")                              
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)

cap = cv2.VideoCapture(0)
print cap.isOpened
ID = 0
ret, img = cap.read() 
while True:
    ret, img = cap.read()                                                   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                                
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                         
    for (x, y, w, h) in faces:                                                  

        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
        gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))               
        
        ID, conf = recognise.predict(gray_face)        
        print conf
        if ID>0:
            if ID==1:
                name='rony'
            elif ID==2:
                name='ayesha'
        else:
           name='unknown' 
        cv2.cv.PutText(cv2.cv.fromarray(img),name, (x,y),font, 255)
    cv2.imshow('EigenFace Face Recognition System', img)                       
    if cv2.waitKey(1) & 0xFF == ord('q'):                                       
        break
cap.release()
cv2.destroyAllWindows()
