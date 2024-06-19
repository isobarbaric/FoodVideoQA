from scipy.spatial import distance as dist
from threading import Thread
import numpy as np
import argparse
import imutils
from imutils import face_utils
import time
import dlib
import cv2
from pathlib import Path


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('ckpts/shape_predictor_68_face_landmarks.dat')

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.79

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)


def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar


def determine_mouth_open(image_path: Path):
	frame = cv2.imread(str(image_path))

	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# print(shape)	
		# print(shape.shape)

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio
		mouth = shape[mStart:mEnd]

		mouthMAR = mouth_aspect_ratio(mouth)
		print(mouthMAR)

		mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)
		
		# Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			print("mouth open")
			return True

	return False


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", required=True, help="Path to image")
	args = parser.parse_args()

	determine_mouth_open(args.path)


if __name__ == "__main__":
    main()