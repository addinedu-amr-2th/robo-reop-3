import cv2
import numpy as np
from rclpy.qos import QoSProfile
import rclpy
from std_msgs.msg import Int32
import time
import pandas as pd

fps = 10

frame_w = 320
frame_h = 240

num_widow = 20
margin = 10
minpix = 25

lower_limit = -50
upper_limit = 50

blue_threshold = 375  #  number of blue pixels thresold
red_threshold = 375  # number of red pixels thresold
intersection_threshold = 48

black_pixels_count_threshold = 20  # Black pixel count threshold
intersection_threshold = 80  # Top Intersection Value
intersection_flag = 0

leftx_current = 0
rightx_current = 0

def display_thresholded_channels(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    _, L, S = cv2.split(frame)
    _, L_thr = cv2.threshold(L, 230, 255, cv2.THRESH_BINARY)
    _, S_thr = cv2.threshold(S, 250, 255, cv2.THRESH_BINARY)

    # Apply thresholding results to HLS images
    frame[:, :, 1] = L_thr
    frame[:, :, 2] = S_thr

    return frame

def apply_perspective_transform(img):
    # Original Perspective
    src_points_image = np.array([         
        (0, 0),
        (frame_w, 0),
        (frame_w, frame_h),
        (0,frame_h) 
        ],dtype=np.float32)

    # a target perspective
    dst_points = np.array([
        (0, 0),
        (frame_w, 0),
        (frame_w, frame_h),
        (0,frame_h)
    ],dtype=np.float32)

    # Calculate the perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points_image, dst_points)
    inv_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points_image)

    # Apply perspective transformation to an image
    transformed_img = cv2.warpPerspective(img, transform_matrix, (frame_w, frame_h))

    return transformed_img, transform_matrix, inv_transform_matrix

def Sliding_window(frame):
    global num_widow, margin, minpix, intersection_threshold, leftx_current, rightx_current

    histogram = np.sum(frame[frame.shape[0]//2:,:], axis=0) 
    midpoint = int(histogram.shape[0] / 2)
    # Returns the index of the largest value in the right and left ranges
    left_max_pexel = np.argmax(histogram[:midpoint]) 
    right_max_pexel = np.argmax(histogram[midpoint:]) + midpoint 
    
    window_height = int(frame.shape[0]/num_widow)
    nz = frame.nonzero() 

    left_lane_inds = []
    right_lane_inds = []

    leftx, lefty, rightx, righty = [], [], [], []

    out_frame = np.dstack((frame, frame, frame))*255

    # Create a sliding window
    for window in range(num_widow):

        win_yl = frame.shape[0] - (window+1)*window_height
        win_yh = frame.shape[0] - window*window_height 

        win_xll = left_max_pexel - margin 
        win_xlh = left_max_pexel + margin 
        win_xrl = right_max_pexel - margin 
        win_xrh = right_max_pexel + margin 

        cv2.rectangle(out_frame,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
        cv2.rectangle(out_frame,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

        left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]

        right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(left_inds)
        right_lane_inds.append(right_inds)

        # Verification of intersections
        window_image = frame[win_yl:win_yh]
        intersection_frame = frame[0:intersection_threshold]

        black_pixels_count = np.sum(intersection_frame == 255)

        if black_pixels_count <= black_pixels_count_threshold:
            intersection_flag = 1
        else:
            intersection_flag = 0

        # Minpix prevents incorrect lane recognition in noisy areas
        # Create the next window with the largest pixels within the previous window X range
        if len(left_inds) > minpix:
            if window_image[win_xll:win_xlh].any():
                leftx_current = np.argmax(window_image[win_xll:win_xlh])
            else:
                leftx_current = int(np.mean(nz[1][left_inds]))

        if len(right_inds) > minpix:
            if window_image[win_xrl:win_xrh].any():
                rightx_current = np.argmax(window_image[win_xrl:win_xrh])
            else:
                rightx_current = int(np.mean(nz[1][right_inds]))

        leftx.append(leftx_current)
        lefty.append((win_yl + win_yh)/2)

        rightx.append(rightx_current)
        righty.append((win_yl + win_yh)/2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    lfit = np.polyfit(np.array(lefty),np.array(leftx),2)
    rfit = np.polyfit(np.array(righty),np.array(rightx),2)

    out_frame[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]


    out_frame[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]

    return lfit, rfit, out_frame, intersection_flag

def draw_lane_lines(original_image, warped_image, transform_matrix, left_fit, right_fit):
    global lower_limit, upper_limit
    yMax = warped_image.shape[0] 
    ploty = np.linspace(0, yMax - 1, yMax) 

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right)) 

    mean_x = np.mean((left_fitx, right_fitx), axis=0)

    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))]) 
    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74)) 

    newwarp = cv2.warpPerspective(color_warp, transform_matrix, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0) 

    # Draw a line in the middle column of the camera frame.
    cv2.line(result, (int(result.shape[1] / 2), 0),(int(result.shape[1] / 2), result.shape[0]), (255, 0, 0), 2)

    # Draw a line in the middle column of both lanes and calculate the distance between the two lines.
    center_col = int((left_fitx[120] + right_fitx[120]) / 2)
    cv2.line(result, (center_col, 0), (center_col, result.shape[0]), (0, 0, 255), 2)
    lane_distance = center_col - int(result.shape[1] / 2)

    # 곡선 알고리즘을 위한 중단 상단의 프레임 차
    right_distance_difference = int(abs(right_fitx[120] - right_fitx[60]))
    left_distance_difference = int(abs(left_fitx[120] - left_fitx[60]))

    if lane_distance < lower_limit:
        lane_distance = lower_limit
    elif lane_distance > upper_limit:
        lane_distance = upper_limit

    return pts_mean, result, lane_distance, right_distance_difference, left_distance_difference

def main():
    # Creating a Video Capture Object
    cap = cv2.VideoCapture(1)

    # Video frame size and FPS settings
    cap.set(cv2.CAP_PROP_FPS, fps)

    qos = QoSProfile(depth=1)
    rclpy.init()
    node = rclpy.create_node('controll_node')
    # Creating a Publisher for ROS topic
    pub_lane_distance = node.create_publisher(Int32, '/lane_distance', qos)
    pub_right_distance_difference = node.create_publisher(Int32, '/right_distance_difference', qos)
    pub_left_distance_difference = node.create_publisher(Int32, '/left_distance_difference', qos)
    pub_intersection_flag = node.create_publisher(Int32, '/intersection_flag', qos)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_resized = cv2.resize(frame, (320, 240))

        # Apply thresholding to Lightness and Saturation channels
        result_video = display_thresholded_channels(frame_resized)

        # Convert color frame to grayscale
        gray_blurred_video = cv2.cvtColor(result_video, cv2.COLOR_BGR2GRAY)
        _, gray_thresh_video = cv2.threshold(gray_blurred_video, 160, 255, cv2.THRESH_BINARY)

        # Perspective Transform applying
        transformed_result_video, inv_transform_matrix_video = apply_perspective_transform(gray_thresh_video) 

        # Sliding window applying
        left_fit, right_fit, out_frame, intersection_flag = Sliding_window(transformed_result_video)

        # Draw lane lines
        lane_distance, right_distance_difference, left_distance_difference = draw_lane_lines(frame_resized, transformed_result_video, inv_transform_matrix_video, left_fit, right_fit )

        # count the number of blue pixels
        num_blue_pixels = (out_frame == [255, 0, 0]).all(axis=2).sum()

        # count the number of red pixels
        num_red_pixels = (out_frame == [0, 0, 255]).all(axis=2).sum()

        # Change lane_distance to -50 if the number of blue pixels is greater than or equal to the reference value
        if num_blue_pixels <= blue_threshold:
            lane_distance = -50

        # Change lane_distance to +50 if the number of red pixels is greater than or equal to the reference value
        if num_red_pixels <= red_threshold:
            lane_distance = +50

        # Publish the lane_distance value to the ROS topic
        msg_lane_distance = Int32()
        msg_lane_distance.data = lane_distance
        pub_lane_distance.publish(msg_lane_distance)

        msg_right_distance_difference = Int32()
        msg_right_distance_difference.data = right_distance_difference
        pub_right_distance_difference.publish(msg_right_distance_difference)

        msg_left_distance_difference = Int32()
        msg_left_distance_difference.data = left_distance_difference
        pub_left_distance_difference.publish(msg_left_distance_difference)

        msg_intersection_flag = Int32()
        msg_intersection_flag.data = intersection_flag
        pub_intersection_flag.publish(msg_intersection_flag)

        # 'q' 키를 누르면 종료
        print('차선 중심과 카메라 중심의 차는', lane_distance, '픽셀 입니다.')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 캡처 객체와 창 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()