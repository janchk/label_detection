from __future__ import print_function
import cv2
# from ar_markers import detect_markers
from auxiiary.detect_mod import detect_markers
import numpy as np
import imutils
from imutils.video import FPS

lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == '__main__':
    print('Press "q" to quit')
    # capture = cv2.VideoCapture("PICT0035.AVI")
    # capture = cv2.VideoCapture("not_fisheyeish.mp4")
    # capture = cv2.VideoCapture("det_track_clear.avi")
    # capture = cv2.VideoCapture("not_fisheyeish_clear.mp4")
    capture = cv2.VideoCapture(0)
    # out = cv2.VideoWriter('det_track_ARTAG_.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30.0, (640, 480))
    # fps = FPS().start()

    _, frame_old = capture.read()
    old_gray = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
    markers = None
    # old_points = np.array([[]], dtype=np.float32)

    # old_points = np.array([[]])

    tracker = cv2.TrackerBoosting_create()

    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
        # frame = cv2.resize(frame, (frame.shape[0]*2,frame.shape[1]*2))
        # print(frame.shape)
    else:
        frame_captured = False

    while frame_captured:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (1, 1), 1)
        # cv2.equalizeHist(frame_gray, frame_gray)
        # frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_RECT, (5, 5), 10)
        frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, (10, 10), 10)
        # out.write(frame)
        frame_big = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
        # print(frame_big.shape)
        if not markers:
            markers = detect_markers(frame_gray)
            for marker in markers:
                # bbox = cv2.boxPoints()
                marker.highlite_marker(frame)

                # old_points = np.array([marker.center], dtype=np.float32 )

                old_points = np.array((marker.contours[0][0], marker.contours[1][0], marker.contours[2][0], marker.contours[3][0], marker.center), dtype=np.float32)

        else:

            new_points, status, errors = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)
            print(errors)

            for point in new_points:
                cv2.circle(frame, tuple(point.ravel()), 5, (255, 0, 0), -1)

            old_gray = frame_gray.copy()
            old_points = new_points
            for error in errors:
                if error[0] > 14 or error[0] == 0:
                    markers = None

            # tracker.init(frame, bbox)

        # frame_captured,


        markers_big = detect_markers(frame_big)
        for marker in markers_big:
            marker.highlite_marker(frame_big)

        cv2.imshow('Test Frame', frame)
        cv2.imshow('Test Frame gray', frame_gray)
        # out.write(frame)
        # cv2.imshow('Test Frame1', frame_big)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()