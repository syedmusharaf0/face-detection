import cv2

face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True :
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(frame,minNeighbors=3)
    for (x,y,h,w) in faces :
        cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),3)
    cv2.imshow('FACE DETECTION',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
