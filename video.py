import cv2
import os
import numpy as np

class Images:

    def __init__(self):
        self.source = '1-1.mp4'
        self.capture = None
        self.read_from_images = None
        self.current_image_index = 0
        self.images_path = 'dataset/images/coin'
        self.images_extension = '.png'
        self.videos_path = 'dataset/videos/'
        self.set_capture()

    def set_capture(self):
        if self.source is not None:
            if isinstance(self.source, str):
                self.capture = cv2.VideoCapture(self.videos_path + self.source)
            elif isinstance(self.source, int):
                self.capture = cv2.VideoCapture(self.source)
            else:
                raise Exception('Передан неверный соурс')
        else:
            self.read_from_images = True


    def get_next_frame(self):
        if self.read_from_images:
            print(os.path.join(self.images_path, str(self.current_image_index) + self.images_extension))
            img = cv2.imread(os.path.join(self.images_path, str(self.current_image_index) + self.images_extension))
            self.current_image_index += 1
            return img, None
        if self.source != 'cut.avi': return self.get_next_frame_default_video()
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.resize(frame, (1280, 480))
            frame = frame[50:350,:]
            height, width, _ = frame.shape
            half_width = width // 2
            left_image = frame[:, :half_width]
            right_image = frame[:, half_width:]
            return left_image, right_image
        raise Exception('Видеопоток окончен либо повреждён')

    def get_next_frame_default_video(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = frame[60:430,20:]
            return frame, None
        raise Exception('Видеопоток окончен либо повреждён')

    def show_image(self, names, imgs):
        print(len(imgs))
        for i, img in enumerate(imgs):
            cv2.imshow(names[i], img)
        cv2.waitKey(0)

    def calculate_input_points_for_perspective_transoform(self, points):
        max_x, max_y = 0, 0
        min_x, min_y = float('inf'), float('inf')
        for point in points:
            max_x = max(max_x, point.pt[1])
            max_y = max(max_y, point.pt[0])
            min_x = min(min_x, point.pt[1])
            min_y = min(min_y, point.pt[0])
        #max_x, max_y, min_x, min_y = max_x + 5, max_y + 5, min_x - 5, min_y - 5
        width = max_x - min_x
        height = max_y - min_y
        return np.float32([[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]]), width, height, (int(min_y), int(min_x)), (int(max_y), int(max_x))

    def perspective_transform(self, img, points):
        input_points, maxWidth, maxHeight, p1, p2= self.calculate_input_points_for_perspective_transoform(points)
        self.output_points_for_perspective_transform = np.float32([[0, 0],
                                 [0, maxHeight - 1],
                                 [maxWidth - 1, maxHeight - 1],
                                 [maxWidth - 1, 0]])
        M = cv2.getPerspectiveTransform(input_points, self.output_points_for_perspective_transform)
        out = cv2.warpPerspective(img, M, (int(maxWidth), int(maxHeight)), flags=cv2.INTER_LINEAR)
        return out

    def draw_keypoint_on_img(self, img, points):
        for point in points:
            img[int(point.pt[1])][int(point.pt[0])] = (0, 255, 0)
        return img

    def get_perpendicular_perspective(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        corners = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
        width = 640
        height = 480
        new_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(corners, new_corners)
        result = cv2.warpPerspective(img, M, (width, height))
        self.show_image(('',), (result,))

    def merge_images(self, image1, image2, image_weight=680,
                     image_height=480):
        stitcher = cv2.Stitcher.create()
        status, stitched_image = stitcher.stitch([image1, image2])
        if image_height is not None or image_weight is not None:
            stitched_image = cv2.resize(stitched_image, (image_weight, image_height))
        if status == cv2.Stitcher_OK:
            return stitched_image

    def save_video_from_frames(self, frames):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1
        frame_size = (640, 480)
        out = cv2.VideoWriter('dataset/result/output.mp4', fourcc, fps, frame_size)
        for frame in frames:
            out.write(frame)
        out.release()

    def rotate_one_image_as_another(self, src_img, target_img):
        angle = 45
        center = (src_img.shape[1] // 2, src_img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(src_img, rotation_matrix, (src_img.shape[1], src_img.shape[0]))
        rotated_img = rotated_img[:target_img.shape[0], :target_img.shape[1]]
        return rotated_img

    def rotated_img(self,img, angle):
        center = (img.shape[1] // 2, img.shape[0] // 2)
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


    def get_next_frame_for_two_cameras(self):
        ret, frame = self.capture.read()
        if ret:
            ##frame = cv2.resize(frame, (640, 480))
            #return frame, None
            frame = cv2.resize(frame, (1280, 480))
            height, width, _ = frame.shape
            half_width = width // 2
            left_image = frame[:, :half_width]
            right_image = frame[:, half_width:]
            return left_image, right_image
        else:
            return None, None
        raise Exception('Видеопоток окончен либо повреждён')