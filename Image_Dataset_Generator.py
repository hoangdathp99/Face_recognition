import cv2
import numpy as np
import os
from math import sqrt
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from constant import IMAGE_SIZE, MAX_NUMBER_OF_IMAGES

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=IMAGE_SIZE)
video_capture = cv2.VideoCapture(0)
name = input("Enter name of person:")

path = 'images'
print(path)
directory = os.path.join(path, name)
print(directory)
if not os.path.exists(directory):
	os.makedirs(directory, exist_ok = 'True')

number_of_images = 0
count = 0

while number_of_images < MAX_NUMBER_OF_IMAGES:
	ret, frame = video_capture.read()

	frame = cv2.flip(frame, 1)

	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	faces = detector(frame_gray)
	for rect in faces:
		x = rect.left()
		y = rect.top()
		w = rect.right()
		h  =rect.bottom()
		cv2.rectangle(frame, (x, y), (x+w,y+h), (0,255,0), 1)
		landmark = shape_predictor(frame_gray, rect)
		lines = []
		for k, d in enumerate(landmark.parts()):
			if(k >= 60 and k <= 68):
				lines.append((d.x,d.y))
		x_line = round((lines[4][0]+lines[0][0])/2)
		y_line = round((lines[4][1]+lines[0][1])/2)

		u_x = (lines[2][0]-x_line)*(lines[2][0]-x_line)
		u_y = (lines[2][1]-y_line)*(lines[2][1]-y_line)

		d_x = (lines[6][0]-x_line)*(lines[6][0]-x_line)
		d_y = (lines[6][1]-y_line)*(lines[6][1]-y_line)
		cv2.rectangle(frame, (x, y), (w,h), (0, 0, 255), 2)
		cv2.rectangle(frame, (x, h - 35), (w, h), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		if sqrt(u_x+u_y) < sqrt(d_x+d_y):
			cv2.putText(frame, 'Happy', (x + 6, h - 6), font, 1.0, (255, 255, 255), 1)
		else:
			if sqrt(u_x+u_y) > sqrt(d_x+d_y):
				if sqrt(u_x+u_y) - sqrt(d_x+d_y) >= 1:
					cv2.putText(frame, 'Sad', (x + 6, h - 6), font, 1.0, (255, 255, 255), 1) 
				else:
					cv2.putText(frame, 'Normal', (x + 6, h - 6), font, 1.0, (255, 255, 255), 1)

    #    cv2.line(frame, (lines[0][0],lines[0][1]) , (lines[4][0],lines[4][1]), (193, 42, 77), 2)
    #    cv2.circle(frame, (x_line, y_line), 5, (0, 0, 255), -1)
	if len(faces) == 1:
		face = faces[0]
		(x, y, w, h) = face_utils.rect_to_bb(face)
		face_img = frame_gray[y-50:y + h+100, x-50:x + w+100]
		face_aligned = face_aligner.align(frame, frame_gray, face)

		if count == 5:
			cv2.imwrite(os.path.join(directory, str(name+str(number_of_images)+'.jpg')), face_aligned)
			number_of_images += 1
			count = 0
		print(count)
		count+=1
		

	cv2.imshow('Video', frame)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

video_capture.release()
cv2.destroyAllWindows()
