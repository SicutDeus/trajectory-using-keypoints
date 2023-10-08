import copy
import sys
import time
import pylab
import numpy as np

from trajectory import Trajectory
from video import Images
import cv2


def main_loop_default_video_front():
    trj = Trajectory()
    video = Images()
    counter = 0
    frames = []
    angle = 0
    while True:
        left_frame, right_frame = video.get_next_frame_for_two_cameras()
        if left_frame is None:
            trj.plott(0,0,0,sleep=True, save=True)
        kp1, des1 = trj.get_point_and_descriptors(left_frame)
        if counter > 0:
            trj.only_one_images_bias(prev_img, left_frame, kp1, kp2, des1, des2)
            cv2.imshow('origin', left_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        counter += 1
        prev_img = left_frame
        kp2 = kp1
        des2 = des1
        if counter == 2000:
            trj.plott(0,0,0,sleep=True, save=True)
            break
    video.save_video_from_frames(frames)


def main_loop_default_video():
    trj = Trajectory()
    video = Images()
    counter = 0
    frames = []
    angle = 0
    while True:
        left_frame, right_frame = video.get_next_frame()
        #left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        if counter > 0:
            kp1, des1 = trj.get_point_and_descriptors(left_frame)
            kp2, des2 = trj.get_point_and_descriptors(prev_img)
            #video.show_image(('1','2'), (prev_img, left_frame))
            try:
                new_angle, new_curr = trj.calculate_angle_between_two_images(prev_img, left_frame, kp1,kp2,des1,des2, angle)
                angle += new_angle
            except Exception as e:
                print(e)
                sys.exit()
                angle = angle
            if 'new_curr' not in locals():
                new_curr = left_frame
            if 'rotated_img' not in locals():
                rotated_img = left_frame
            rotated_img = video.rotated_img(left_frame, angle)
            trj.only_one_images_bias(prev_rotated, rotated_img, kp1, kp2, des1, des2)
            cv2.imshow('origin', left_frame)
            cv2.imshow('rotated', rotated_img)
            cv2.imshow('new_rotated', new_curr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        counter += 1
        prev_img = left_frame
        if 'rotated_img' not in locals():
            rotated_img = left_frame
        else:
            prev_rotated = rotated_img
        if 'prev_rotated' not in locals():
            prev_rotated = rotated_img
    video.save_video_from_frames(frames)

if __name__ == '__main__':
    #main_loop_default_video()
    main_loop_default_video_front()
    #main_loop_default_video()
    #check_merge()
    #main()