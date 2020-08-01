import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
profileface_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

capture = cv2.VideoCapture(0)

while True 	:
	_, img = capture.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.1, 2)

	eye= eye_cascade.detectMultiScale(gray,1.5,2)

	profileface = profileface_cascade.detectMultiScale(gray, 1.1, 3)

	smile = smile_cascade.detectMultiScale(gray,1.1,30)


	for(x,y,w,h) in eye:
		cv2.rectangle(img, (x,y) , (x+w, y+h), (38,226,240), 1)

	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x,y) , (x+w, y+h), (0,255,0), 1)

	for(x,y,w,h) in smile:
		cv2.rectangle(img, (x,y) , (x+w, y+h), (0,0,255), 1)

	cv2.imshow('img',img)
	k= cv2.waitKey(30) & 0xff
	if k==27:
		break

capture.release() 
