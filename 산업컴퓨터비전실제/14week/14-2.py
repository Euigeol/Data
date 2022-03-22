import cv2
import numpy as np

def detect_faces(video_file, detector, win_title):
    cap = cv2.VideoCapture(video_file)
    
    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 255, 0), 3)
            text_size, _ = cv2.getTextSize('Face', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (x, y - text_size[1]), (x +text_size[0], y), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, 'Face', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) 

        cv2.imshow(win_title, frame)

        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

haar_face_cascade = cv2.CascadeClassifier('c:/study/14week/haarcascade_frontalface_default.xml')
detect_faces('c:/study/14week/mo.mp4', haar_face_cascade, 'haar cascade face detector')

lbp_face_cascade = cv2.CascadeClassifier()
lbp_face_cascade.load('c:/study/14week/lbpcascade_frontalface.xml')

detect_faces(0, lbp_face_cascade, 'LBP cascade face detector')
    
    

     
     

image = cv2.imread('c:/study/14week/image.jpg')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

locations, weight = hog.detectMultiScale(image)

dbg_image = image.copy()
for loc in locations:
    cv2.rectangle(dbg_image, (loc[0], loc[1]),
                  (loc[0]+loc[2], loc[1]+loc[3]), (0, 255, 0), 2)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(image[:,:,[2,1,0]]) 
plt.subplot(122)
plt.title('detectios')
plt.axis('off')
plt.imshow(dbg_image[:,:,[2,1,0]]) 
plt.tight_layout()
plt.show()  