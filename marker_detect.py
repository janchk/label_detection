from __future__ import print_function
import cv2
from ar_markers import detect_markers
import imutils
from imutils.video import FPS

if __name__ == '__main__':
        print('Press "q" to quit')
        # capture = cv2.VideoCapture("PICT0035.AVI")qq
        capture = cv2.VideoCapture(0)
        # out = cv2.VideoWriter('not_fisheyeish.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640, 480))
        # fps = FPS().start()

        if capture.isOpened():  # try to get the first frame
                frame_captured, frame = capture.read()
                # frame = cv2.resize(frame, (frame.shape[0]*2,frame.shape[1]*2))
                # print(frame.shape)
        else:
                frame_captured = False

        while frame_captured:
                # out.write(frame)
                frame_big = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
                # print(frame_big.shape)
                markers = detect_markers(frame)
                for marker in markers:
                        marker.highlite_marker(frame)

                markers_big = detect_markers(frame_big)
                for marker in markers_big:
                        marker.highlite_marker(frame_big)

                cv2.imshow('Test Frame', frame)
                # out.write(frame)
                cv2.imshow('Test Frame1', frame_big)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
                frame_captured, frame = capture.read()

        # When everything done, release the capture
        capture.release()
        cv2.destroyAllWindows()