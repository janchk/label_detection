from __future__ import print_function
import cv2
import cv2.xfeatures2d
from ar_markers import detect_markers
import numpy as np
import imutils
from imutils.video import FPS


def descr_computing(img):

    # detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    detector = cv2.xfeatures2d.MSDDetector_create()
    # star = cv2.xfeatures2d.StarDetector_create()
    freak = cv2.xfeatures2d.FREAK_create()

    kpd = detector.detect(img, None)

    kp, descr = freak.compute(img, kpd)
    return kpd, kp, descr


if __name__ == '__main__':
    img_1 = cv2.imread("marker_2.png", 0)
    img_2 = cv2.imread("vlcsnap.png", 0)

    kpd1, kp1, descr1 = descr_computing(img_1)
    kpd2, kp2, descr2 = descr_computing(img_2)
    print(descr1.shape)
    print(descr2.shape)

    img_with_kpd1 = cv2.drawKeypoints(img_1, kpd1, img_1, color=(12, 200, 1), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    img_with_kpd2 = cv2.drawKeypoints(img_2, kpd2, img_2, color=(12, 200, 1), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # print(freak.getInt('bytes'))

    #FEATURE MATCHING
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # knn_matches = flann.knnMatch(descr1, descr2, k=2, compactResult=True)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descr1, descr2)

    ratio_thresh = 0.7
    good_matches = []
    # for m, n in matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)

    matches = sorted(matches, key=lambda x: x.distance)
    #-- Draw matches
    img_matches = np.empty((max(img_1.shape[0], img_2.shape[0]), img_1.shape[1]+img_2.shape[1], 3), dtype=np.uint8)
    # cv2.drawMatches(img_1, kp1, img_2, kp2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_3 = cv2.drawMatches(img_1, kp1, img_2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    while True:
        cv2.imshow('Test Frame1', img_3)
        cv2.imshow('Test Frame_kpd1', img_with_kpd1 )
        cv2.imshow('Test Frame_kpd2', img_with_kpd2 )
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # # capture = cv2.VideoCapture("PICT0035.AVI")qq
    # # out = cv2.VideoWriter('not_fisheyeish.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (640, 480))
    # # fps = FPS().start()
    #
    # if capture.isOpened():  # try to get the first frame
    #         frame_captured, frame = capture.read()
    #         # frame = cv2.resize(frame, (frame.shape[0]*2,frame.shape[1]*2))
    #         # print(frame.shape)
    # else:
    #         frame_captured = False
    #
    # while frame_captured:
    #         # out.write(frame)
    #         frame_big = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
    #         # print(frame_big.shape)
    #         markers = detect_markers(frame)
    #         for marker in markers:
    #                 marker.highlite_marker(frame)
    #
    #         markers_big = detect_markers(frame_big)
    #         for marker in markers_big:
    #                 marker.highlite_marker(frame_big)
    #
    #         cv2.imshow('Test Frame', frame)
    #         # out.write(frame)
    #         cv2.imshow('Test Frame1', frame_big)
    #         if cv2.waitKey(20) & 0xFF == ord('q'):
    #                 break
    #         frame_captured, frame = capture.read()
    #
    # # When everything done, release the capture
    # capture.release()
    # cv2.destroyAllWindows()
