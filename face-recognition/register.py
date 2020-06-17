# import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import sys

#  the arguments
cascPath = sys.argv[1]

currDir = os.getcwd()
outputDir = os.path.join(currDir,'FACES') 

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

targetDir = os.path.join(currDir,cascPath) 

print("[INFO] Files will be saved in : " + targetDir)

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# prepare folder
if not os.path.exists(targetDir):
    os.makedirs(targetDir)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	orig = frame.copy()
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))
	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

    #Save just the rectangle faces in SubRecFaces
	sub_face = orig[y:y+h, x:x+w]

 
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		fileName = "/{}.png".format(str(total).zfill(5))        
		p = targetDir + fileName
		cv2.imwrite(p, sub_face)
		total += 1
	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()