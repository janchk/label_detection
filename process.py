import cv2
import pyrealsense2 as rs
import numpy as np
from apriltag_detector import apriltag_detector

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('det_apriltag_rs_fisheye.mp4', fourcc, 30.0, (1696, 1600), False)
pipeline = rs.pipeline()
config = rs.config()

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        # fisheye_frame = frames.get_fisheye_frame()
        fisheye_frame_1 = np.asanyarray(frames.get_fisheye_frame(1).get_data())
        fisheye_frame_2 = np.asanyarray(frames.get_fisheye_frame(2).get_data())

        frame_res_1, data_1 = apriltag_detector(fisheye_frame_1)
        frame_res_2, data_2 = apriltag_detector(fisheye_frame_2)

        stereo_pair = np.hstack((fisheye_frame_1, fisheye_frame_2))
        res_pair = np.hstack((frame_res_1, frame_res_2))
        demo_img = np.vstack((stereo_pair, res_pair))

        cv2.namedWindow('RealSense_apriltag', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense_apriltag', demo_img)
        out.write(demo_img)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', fisheye_frame)
        # cv2.imshow('RealSense', images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:

    pipeline.stop()