import cv2
import sys
sys.path.insert(1, "/media/jakhremchik/ubuntuHD/PROJECTS/DRONE/apriltags3-py/")
import apriltag


def apriltag_detector(frame):
    """
    detector for markers using 'apriltag' system
    :param frame:
    :return: img, data
    """
    if len(frame.shape) > 2:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    frame_gray = cv2.GaussianBlur(frame_gray, (1, 1), 1)

    # cv2.equalizeHist(frame_gray, frame_gray)
    frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, (10, 10), 10)

    detector = apriltag.Detector()
    data, frame_res = detector.detect(frame_gray, return_image=True)

    return frame_res, data


if __name__ == '__main__':
    print('Press "q" to quit')
    # capture = cv2.VideoCapture("det_track_clear_apriltag_.avi")
    capture = cv2.VideoCapture(0)
    # out = cv2.VideoWriter('det_qqtrack.avi', cv2.VideoWriter_fourcc(*'MPEG'), 20.0, (640, 480))

    if capture.isOpened():  # try to get the first frame
        frame_captured, frame = capture.read()
    else:
        frame_captured = False

    while frame_captured:
        frame_res, data = apriltag_detector(frame)

        cv2.imshow('Test Frame', frame)
        cv2.imshow('Test Frame result', frame_res)
        # out.write(frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        frame_captured, frame = capture.read()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
