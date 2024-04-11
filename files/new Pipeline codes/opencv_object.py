import cv2
from imutils.video import VideoStream
from imutils.video import FPS

# OPENCV_OBJECT_TRACKERS = {
# 		"csrt": cv2.TrackerCSRT_create,
# 		"kcf": cv2.TrackerKCF_create,
# 		"boosting": cv2.TrackerBoosting_create,
# 		"mil": cv2.TrackerMIL_create,
# 		"tld": cv2.TrackerTLD_create,
# 		"medianflow": cv2.TrackerMedianFlow_create,
# 		"mosse": cv2.TrackerMOSSE_create
# 	}


tracker = cv2.TrackerKCF_create()
initBB = None
fps = None
vs = cv2.VideoCapture(0)
while True:
	frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]
    if initBB is not None:
        
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        fps.update()
        fps.stop()
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
            tracker.init(frame, initBB)
            fps = FPS().start()
        elif key == ord("q"):
            break


cv2.destroyAllWindows()