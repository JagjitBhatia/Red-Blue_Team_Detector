import cv2
import numpy as np
from PIL import Image
import math
from imutils.object_detection import non_max_suppression

# Returns True if rectangle A (defined by topLeft_A and bottomRight_A)
# is inside rectangle B (defiend by topLeft_B and bottomRight_B) by a 10% margin
# ~~ topLeft_X and bottomRight_X are tuples of (x,y) ~~
def isInside(topLeft_A, bottomRight_A, topLeft_B, bottomRight_B):
	return (topLeft_A[0] * 1.10) >= topLeft_B[0] and bottomRight_A[1] <= (bottomRight_B[1] * 1.10)

# Returns a new list of rectangles that contains no nested rectangles
# A rectangle is a tuple of (xA, yA, xB, yB), where A is the top left corner
# of a rectangle and B is the bottom right corner
def filterNested(rects):
	new_rects = []

	for R1 in rects:
		nested = False
		for R2 in rects:
			if tuple(R1) == tuple(R2):
				continue
			if isInside((R1[0], R1[1]), (R1[2], R1[3]), (R2[0], R2[1]), (R2[2], R2[3])):
				nested = True
				break
		if nested == False:
			new_rects.append(R1)
	
	return new_rects
		

	

def MAX_COLOR(img, topLeft, bottomRight):
	

	colors = {}
	ORIGIN = (round((bottomRight[0] - topLeft[0])/2) + topLeft[0], round((bottomRight[1] - topLeft[1])/2) + topLeft[1])
	
	for i in range(topLeft[0], bottomRight[0] + 1):   # x iterator
		#print("I: ", i)
		for j in range(topLeft[1] + 1, bottomRight[1]):   # y iterator
			if i < 0 or i >= img.shape[1] or j < 0 or j >= img.shape[0]:
				continue
			#print("J: ", j)
			dist = round(math.sqrt(abs(ORIGIN[0]-i))**2 + abs((ORIGIN[1]-j)**2))  # distance magnitude
			strength = 1/(1 + dist)  # closeness to origin = strength

			color_key = str(img[j][i][0]) + ":" + str(img[j][i][1]) + ":" + str(img[j][i][2])


			#print(color_key)
			if color_key in colors:
				colors[color_key] += strength
			else:
				colors[color_key] = strength
	
	# colors_sorted = dict(sorted(colors.items(), key=lambda item: item[1], reverse=True))
	# f = open('colorDetector.txt', 'w')
	# f.write(str(colors_sorted))

	max_color = max(colors, key=colors.get)

	return max_color

def IS_BLUE(max_color_key):
	max_color_str_vector = max_color_key.split(':')
	
	red = int(max_color_str_vector[2])
	green = int(max_color_str_vector[1])
	blue = int(max_color_str_vector[0])

	return (blue > 100 and blue > green and (blue + green) > red * 2)


def IS_RED(max_color_key):
	max_color_str_vector = max_color_key.split(':')
	
	red = int(max_color_str_vector[2])
	green = int(max_color_str_vector[1])
	blue = int(max_color_str_vector[0])

	return (red > 100 and red > green * 2 and red > blue * 2)




# Enable camera
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

snapshot_counter = 0

done = False

red = False
blue = False


misses = 0

while True:
	success, img = cap.read()
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Getting corners around the face
	faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor

	bodies = [] 

	if len(faces) == 0:
		bodies, weights = hog.detectMultiScale(imgGray, winStride=(8,8))
		bodies = non_max_suppression(bodies, probs=None, overlapThresh=0.001)
		bodies = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bodies])
		bodies = filterNested(bodies)
	
	font = cv2.FONT_HERSHEY_DUPLEX

	if red == True:
		cv2.putText(img, 'Red Identified!', (14, 25), font, 1, (0, 0, 255), 2)
	elif blue == True:
		cv2.putText(img, 'Blue Identified!', (14, 25), font, 1, (255, 0, 0), 2)
	else:
		cv2.putText(img,'Identifying...',(14, 25), font, 1,(255, 0, 255), 2)

	

	# drawing bounding box around face
	for (x, y, w, h) in faces:
		#img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
		snapshot_counter += 1
		headWidth = 2 * w 
		torsoHeight = 4 * h
		leftEdge = x - w
		rightEdge = x + headWidth
		height = y + torsoHeight
		img = cv2.rectangle(img, (int(leftEdge), y), (int(rightEdge), int(height)), (0, 255, 0), 3)
		topLeft = (int(leftEdge), y)
		bottomRight = (int(rightEdge), int(height))

		
    
		if snapshot_counter == 20:
			max_color_key = MAX_COLOR(img, (int(leftEdge), y), (int(rightEdge), int(height)))
			captured = ''

			if IS_BLUE(max_color_key) == True:
				captured = 'blue'
				if red == True:
					blue = True
					red = False
			
			elif IS_RED(max_color_key) == True:
				captured = 'red'
				if blue == True:
					blue = False
					red = True

			else:
				captured = 'none'

			if blue == True and (captured == 'none'):
				misses += 1
			
			if red == True and (captured == 'none'):
				misses += 1

			

			if misses > 3 or (blue == False and red == False):
				if captured == 'blue':
					blue = True
					red = False
				
				elif captured == 'red':
					blue = False
					red = True
				
				else:
					blue = False
					red = False
				
				misses = 0

			snapshot_counter = 0
				
				
			
			
				

			
			
		#exit()

	
		img = cv2.rectangle(img, (int(leftEdge), y), (int(rightEdge), int(height)), (0, 255, 0), 3)
	
	for (xA, yA, xB, yB) in bodies:
		snapshot_counter += 1

		if snapshot_counter == 20:
			max_color_key = MAX_COLOR(img, (xA, yA), (xB, yB))
			print('MAX COLOR: ', max_color_key)
			captured = ''

			if IS_BLUE(max_color_key) == True:
				captured = 'blue'
				if red == True:
					blue = True
					red = False
			
			elif IS_RED(max_color_key) == True:
				captured = 'red'
				if blue == True:
					blue = False
					red = True

			else:
				captured = 'none'

			if blue == True and (captured == 'none'):
				misses += 1
			
			if red == True and (captured == 'none'):
				misses += 1

			

			if misses > 3 or (blue == False and red == False):
				if captured == 'blue':
					blue = True
					red = False
				
				elif captured == 'red':
					blue = False
					red = True
				
				else:
					blue = False
					red = False
				
				misses = 0

			snapshot_counter = 0
				
		# display the detected boxes 
		cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 255), 2)

	cv2.imshow('face_detect', img)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyWindow('face_detect')