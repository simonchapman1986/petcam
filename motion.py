# USAGE
# python motion.py --conf conf.json

from pyimagesearch.tempimage import TempImage
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON config")
args = vars(ap.parse_args())

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

DETECTED = "Pet on the move"
EMPTY = "Pet Sleeping"

# check cam setup
if conf["pi_cam"]:
	# For use with Raspberry pi camera
	from picamera.array import PiRGBArray
	from picamera import PiCamera
	camera = PiCamera()
	camera.resolution = tuple(conf["resolution"])
	camera.framerate = conf["fps"]
	rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))
else:
	# For use with web camera
	camera = cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# allow the cam to warmup
print("warming up camera...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motion_counter = 0


# capture frames from the cam
while camera.isOpened():
	ret, frame = camera.read()
	if ret:
		# write frame
		out.write(frame)

		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

	timestamp = datetime.datetime.now()
	text = EMPTY

	# resize the frame
	# convert it to grayscale
	# blur the frame
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the average frame is None, initialize it
	if avg is None:
		print("starting bg model...")
		avg = gray.copy().astype("float")
		continue

	# accumulate weighted avg between the frame and
	# last frames, then compute the diff between the
	# frame and average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# threshold the delta, dilate the image to fill
	# in holes, then find contours on the image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	for c in cnts:
		# if the contour is small just ignore
		if cv2.contourArea(c) < conf["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = DETECTED

	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

	# check to see if pet has moved
	if text == DETECTED:
		# check to see if enough time has passed
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motion_counter += 1

			# check to see if the number of frames with consistent motion is
			# high enough
			if motion_counter >= conf["min_motion_frames"]:
				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motion_counter = 0

	# otherwise, the pet has not moved
	else:
		motionCounter = 0

	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the petCam feed
		cv2.imshow("PetCam Feed", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break

	# clear for the next frame
	if conf["pi_cam"]:
		rawCapture.truncate(0)

# Release everything if job is finished
camera.release()
out.release()
cv2.destroyAllWindows()