import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

# GPIO Settings
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pin setting
motorR_forward = 23
motorR_backward = 24
motorL_forward = 25
motorL_backward = 26

# Pin Settings
GPIO.setup(motorR_forward, GPIO.OUT)
GPIO.setup(motorR_backward, GPIO.OUT)
GPIO.setup(motorL_forward, GPIO.OUT)
GPIO.setup(motorL_backward, GPIO.OUT)

# PWM 설정
pwmR = GPIO.PWM(motorR_forward, 100)  # 100Hz
pwmL = GPIO.PWM(motorL_forward, 100)  # 100Hz

# PWM 시작 (Duty Cycle = 0)
pwmR.start(0)
pwmL.start(0)

# P-control constant
Kp = 0.2

leftx_current = 0
rightx_current = 0

fps = 10

frame_w = 320
frame_h = 240

num_widow = 20
margin = 10
minpix = 25

lower_limit = -50
upper_limit = 50


# def display_thresholded_channels(frame):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
#     _, L, S = cv2.split(frame)
#     _, L_thr = cv2.threshold(L, 230, 255, cv2.THRESH_BINARY)
#     _, S_thr = cv2.threshold(S, 250, 255, cv2.THRESH_BINARY)

#     # Apply thresholding results to HLS images
#     frame[:, :, 1] = L_thr
#     frame[:, :, 2] = S_thr

#     return frame

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

    return lfit, rfit, out_frame

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

    # Frame difference at the top of the break for curved algorithms
    right_distance_difference = int(abs(right_fitx[120] - right_fitx[60]))
    left_distance_difference = int(abs(left_fitx[120] - left_fitx[60]))

    if lane_distance < lower_limit:
        lane_distance = lower_limit
    elif lane_distance > upper_limit:
        lane_distance = upper_limit

    return pts_mean, result, lane_distance, right_distance_difference, left_distance_difference

def control_motor(lane_distance):
    if lane_distance > 0: 
        right_speed = 100 - Kp * lane_distance
        left_speed = 100
    else:
        right_speed = 100
        left_speed = 100 + Kp * lane_distance

    # 모터 속도 설정
    pwmR.ChangeDutyCycle(right_speed)
    pwmL.ChangeDutyCycle(left_speed)


def main():
    # Creating a Video Capture Object
    cap = cv2.VideoCapture(0)

    # Video frame size and FPS settings
    cap.set(cv2.CAP_PROP_FPS, fps)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_resized = cv2.resize(frame, (320, 240))

        # Apply thresholding to Lightness and Saturation channels
        # result_video = display_thresholded_channels(frame_resized)

        # Convert color frame to grayscale
        gray_blurred_video = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        _, gray_thresh_video = cv2.threshold(gray_blurred_video, 160, 255, cv2.THRESH_BINARY)

        # Perspective Transform applying
        transformed_result_video, transform_matrix, inv_transform_matrix_video = apply_perspective_transform(gray_thresh_video) 

        # Sliding window applying
        left_fit, right_fit, outframe = Sliding_window(transformed_result_video)

        # Draw lane lines
        lane_distance = draw_lane_lines(frame_resized, transformed_result_video, inv_transform_matrix_video, left_fit, right_fit )

        cv2.imshow('result_video', outframe)
        cv2.imshow('frame_resized', frame_resized)
        cv2.imshow('transformed_result_video', transformed_result_video)
        cv2.imshow('gray_thresh_video', gray_thresh_video)
        # cv2.imshow('result_video', result_video)
        
        # Press the 'q' key to end
        print('차선 중심과 카메라 중심의 차는', lane_distance, '픽셀 입니다.')

        control_motor(lane_distance)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()