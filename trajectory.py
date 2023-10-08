import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from plot import DynamicUpdate
from pprint import pprint


class Trajectory:
    def __init__(self):
        akaze = cv2.AKAZE_create()
        sift = cv2.SIFT_create()
        orb = cv2.ORB_create()
        brisk = cv2.BRISK_create()
        detector_type = 'ORB'

        self.detector = None
        if detector_type == 'AKAZE':
            self.detector = akaze
        elif detector_type == 'SIFT':
            self.detector = sift
        elif detector_type == 'ORB':
            self.detector = orb
        elif detector_type == 'BRISK':
            self.detector = brisk

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.curr_x, self.curr_y = 0, 0
        self.current_offset_x, self.current_offset_y = 0, 0
        plt.ion()
        self.plott = DynamicUpdate()
        #self.plott.on_launch()
        self.current_angle_offset = 0
        self.current_angle = 0
        self.distance_in_meters = 0
        self.height = 100
        self.total_x_meters = 0
        self.total_y_meters = 0
        self.total_distance_2 = 0

    def get_point_and_descriptors(self, img):
        kpts, desc = self.detector.detectAndCompute(img, None)
        return kpts, desc

    def left_and_right_images_bias(self, prev_left_img, prev_right_img, curr_left_img, curr_right_img):
        kp1, des1 = self.get_point_and_descriptors(prev_left_img)
        kp2, des2 = self.get_point_and_descriptors(curr_left_img)
        offset_x_1, offset_y_1 = self.calculate_offset(prev_left_img, kp1, kp2, des1, des2)
        kp1, des1 = self.get_point_and_descriptors(prev_right_img)
        kp2, des2 = self.get_point_and_descriptors(curr_right_img)
        offset_x_2, offset_y_2 = self.calculate_offset(prev_right_img, kp1, kp2, des1, des2)

        self.current_offset_x = (offset_x_1 + offset_x_2) / 2
        self.current_offset_y = (offset_y_1 + offset_y_2) / 2

        print('Смещение по оси X:', self.current_offset_x)
        print('Смещение по оси Y:', self.current_offset_y)

        self.add_new_points_on_map()

    def only_one_images_bias(self, prev_img, curr_img, kp1,kp2,des1,des2):
        dist_thres = 1
        offset_x, offset_y = self.calculate_offset(prev_img, kp1, kp2, des1, des2)
        self.current_offset_x = offset_x
        self.current_offset_y = offset_y
        #if -.1 < self.current_offset_x < 1:
        #    self.current_offset_x = 0
        #if -1 < self.current_offset_y < 1:
        #    self.current_offset_y = 0
        dx_meters = (self.height * self.current_offset_x) / prev_img.shape[1]
        dy_meters = (self.height * self.current_offset_y) / prev_img.shape[0]
        total_dist_meters = math.sqrt(dx_meters ** 2 + dy_meters ** 2)
        self.current_offset_x = dx_meters
        self.current_offset_y = dy_meters
        print(f'X метры: {dx_meters}')
        print(f'Y метры: {dy_meters}')
        print(f'Общие метры: {total_dist_meters}')
        self.distance_in_meters += total_dist_meters
        print(f'Общие метры тотал: {self.distance_in_meters}')
        self.total_x_meters += dx_meters
        self.total_y_meters += dy_meters
        total_dist_meters_2 = math.sqrt(self.total_x_meters ** 2 + self.total_y_meters ** 2)
        print(f'Общие метры тотал2: {total_dist_meters_2}')
        print(f'Общее смещение по X: ', self.total_x_meters)
        print(f'Общее смещение по Y: ', self.total_y_meters)
        #print('Смещение по оси X:', self.current_offset_x)
        #print('Смещение по оси Y:', self.current_offset_y)
        #print('Угол смещения: ', self.current_angle_offset)
        self.add_new_points_on_map()

    def angle_between_lines(self, line1_start, line1_end, line2_start, line2_end):
        dx1 = line1_end[0] - line1_start[0]
        dy1 = line1_end[1] - line1_start[1]
        dx2 = line2_end[0] - line2_start[0]
        dy2 = line2_end[1] - line2_start[1]

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)

        return math.degrees(angle2 - angle1)

    def filter_matches(self, kpts1, kpts2, desc1, desc2):
        homography = np.identity(3)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(desc1, desc2, 2)
        matched1 = []
        matched2 = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                matched1.append(kpts1[m.queryIdx])
                matched2.append(kpts2[m.trainIdx])
        inliers1 = []
        inliers2 = []
        good_matches = []
        inlier_threshold = 5  # Distance threshold to identify inliers with homography check
        for i, m in enumerate(matched1):
            col = np.ones((3, 1), dtype=np.float64)
            col[0:2, 0] = m.pt
            col = np.dot(homography, col)
            col /= col[2, 0]
            dist = math.sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                             pow(col[1, 0] - matched2[i].pt[1], 2))
            if dist < inlier_threshold:
                good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
                inliers1.append(matched1[i])
                inliers2.append(matched2[i])
        return good_matches

    def find_affine(self, src_pts, dst_pts):
        M = cv2.estimateAffine2D(src_pts, dst_pts)[0]
        return M

    def calculate_angle_between_two_images(self, prev_img, curr_img, kp1, kp2, des1, des2, curr_angle):
        center_point = (prev_img.shape[1] // 2, prev_img.shape[0] // 2)
        center_line = ((0, prev_img.shape[0] // 2), (prev_img.shape[1], prev_img.shape[0] // 2))
        angles_interval = []
        for interval in range(-180,178,3):
            angles_interval.append([interval, interval + 3])
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:10]
        src_pts = ([kp1[m.queryIdx].pt for m in matches])
        dst_pts = ([kp2[m.trainIdx].pt for m in matches])
        src_lines = [(center_point, (int(src_pt[0]), int(src_pt[1]))) for src_pt in src_pts]
        dst_lines = [(center_point, (int(dst_pt[0]), int(dst_pt[1]))) for dst_pt in dst_pts]
        center_line_start = center_line[0]
        center_line_end = center_line[1]
        angles_first_img = []
        for line in src_lines:
            angle = self.angle_between_lines(center_line_start, center_line_end, line[0], line[1])
            angles_first_img.append(angle)
        angles_second_img = []
        for line in dst_lines:
            angle = self.angle_between_lines(center_line_start, center_line_end, line[0], line[1])
            angles_second_img.append(angle)
        result_angles = []
        for i, _ in enumerate(angles_first_img):
            result_angles.append(angles_second_img[i] - angles_first_img[i])
        agles = {}
        for angle in result_angles:
            for i, angle_interval in enumerate(angles_interval):
                if angle_interval[0] < abs(angle) <= angle_interval[1]:
                    if f'{angle_interval[0]}-{angle_interval[1]}' not in agles.keys():
                        agles[f'{angle_interval[0]}-{angle_interval[1]}'] = []
                    agles[f'{angle_interval[0]}-{angle_interval[1]}'].append(angle)
                    break

        best_interval = ''
        max_elems = 0
        for key in agles.keys():
            if len(agles[key]) > max_elems:
                max_elems = len(agles[key])
                best_interval = key
        angle_degrees = sum(agles[best_interval])/len(agles[best_interval])
        angle_radians = math.radians(angle_degrees)
        pts1 = np.float32(src_pts).reshape(-1, 1, 2)
        pts2 = np.float32(dst_pts).reshape(-1, 1, 2)
        M = self.find_affine(pts1, pts2)
        M = np.vstack([M, [0., 0., 1.]])
        M = np.linalg.inv(M)
        rot_mat = cv2.getRotationMatrix2D((curr_img.shape[1]//2, curr_img.shape[0]//2), curr_angle+angle_degrees, scale=1)
        warp_rotate_dst = cv2.warpAffine(curr_img, rot_mat, (curr_img.shape[1], curr_img.shape[0]))
        self.current_angle_offset = angle_degrees
        self.current_angle = self.current_angle + self.current_angle_offset
        self.current_angle = math.radians(self.current_angle)



        return -angle_degrees, warp_rotate_dst

    def get_pts(self, kp1,kp2,des1,des2):
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return src_pts, dst_pts

    def calculate_offset(self, prev_img, kp1, kp2, des1, des2):
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, c = prev_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        offset_x = dst[0][0][0] - pts[0][0][0]
        offset_y = dst[0][0][1] - pts[0][0][1]
        return offset_x, offset_y

    def add_new_points_on_map(self):
        if self.current_offset_x != 0 or self.current_offset_y != 0 or True:
            self.curr_x = self.curr_x + self.current_offset_x
            self.curr_y = self.curr_y + self.current_offset_y
            self.plott(self.curr_x, self.curr_y, 100)


