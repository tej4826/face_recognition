import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('elon3.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)

face_locTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(face_locTest[3],face_locTest[0]),(face_locTest[1],face_locTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
# print(results)

# cv2.imshow('img1',imgElon)
# cv2.imshow('img2',imgTest)
# cv2.waitKey(0)