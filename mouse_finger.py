import cv2
import pyautogui
from pynput.mouse import Controller, Button
import math
import time
import os

# mediapipe thư viện để xác định vị trí trên cơ thể hoặc vật gì đó đại khái là xác định vật 

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker,RunningMode

# Đây là fingermouse sử dụng ngón cái để điều khiển vị trí ngón cái + ngón trỏ = left click và ngón trỏ + ngón giữa = right click
# dự án này có chút hỗ trợ bỡi trí tuyện nhân tạo chạy trên python 3.10 và "pip install mediapipe==0.10.9" cái này còn nhiều lỗi lắm thông cảm :v
# à hiện tại muốn bật tắt thì bấm vào file và tắt thì vào console khi đã bấm file rồi tắt ad lười làm cái bản điwwù khiển quá


this_file = os.path.dirname(__file__)
link_model = os.path.join(this_file,"hand_landmarker.task") # lấy đường link file chứa bản mã về xác định ngón tay


# Lấy kích thước màn hình

screen_w, screen_h = pyautogui.size()


# Biến chuột

mouse = Controller()

smooth_mouse_x = None
smooth_mouse_y = None

right_clicked = False
thumb_index_pinch_start = None
left_held = False

HOLD_THRESHOLD = 0.4


# MediaPipe TASKS API

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=link_model),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)


# Webcam

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    frame_timestamp += 1

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    thumb_index_pinch = False
    middle_index_pinch = False

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        def pt(pointer):
            lm = hand[pointer]
            return int(lm.x * w), int(lm.y * h)

        thumb_x, thumb_y = pt(4)
        index_x, index_y = pt(8)
        middle_x, middle_y = pt(12)

        cv2.circle(frame,(thumb_x,thumb_y),8,(0,255,255),-1)
        cv2.circle(frame,(index_x,index_y),8,(0,0,255),-1)
        cv2.circle(frame,(middle_x,middle_y),8,(255,0,0),-1)

        
        # Chỉnh tốc độ chuột
        
        speed_mouse = 4 # chỉnh ở đây
        raw_mouse_x = int(thumb_x * screen_w * speed_mouse / w) - speed_mouse * 500
        raw_mouse_y = int(thumb_y * screen_h * speed_mouse / h) - speed_mouse * 500

        alpha = 0.15 # chống rung con trỏ chuột giá trị càng nhỏ thì nó cành ít rung và bị chậm

        if smooth_mouse_x is None:
            smooth_mouse_x = raw_mouse_x
            smooth_mouse_y = raw_mouse_y
        else:
            smooth_mouse_x = int(alpha * raw_mouse_x + (1-alpha) * smooth_mouse_x)
            smooth_mouse_y = int(alpha * raw_mouse_y + (1-alpha) * smooth_mouse_y)

        dx = abs(smooth_mouse_x - mouse.position[0])
        dy = abs(smooth_mouse_y - mouse.position[1])

        # Điều khiển con trỏ chuột

        if dx > 5 or dy > 5:
            mouse.position = (smooth_mouse_x, smooth_mouse_y)

        # khoản cách

        def dist(p1,p2):
            return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

        thumb_pt=(thumb_x,thumb_y)
        index_pt=(index_x,index_y)
        middle_pt=(middle_x,middle_y)

        d_thumb_index = dist(thumb_pt,index_pt)
        d_middle_index = dist(middle_pt,index_pt)

        pinch_threshold = 0.04 * w

        if d_thumb_index < pinch_threshold:
            thumb_index_pinch = True

        if d_middle_index < pinch_threshold:
            middle_index_pinch = True


        now = time.time()

       
        # LEFT CLICK HOLD Giữ chuột trái
        
        if thumb_index_pinch:

            if thumb_index_pinch_start is None:
                thumb_index_pinch_start = now

            duration = now - thumb_index_pinch_start

            if duration >= HOLD_THRESHOLD and not left_held:
                mouse.press(Button.left)
                left_held = True

        else:

            if thumb_index_pinch_start is not None:

                duration = now - thumb_index_pinch_start

                if duration < HOLD_THRESHOLD:
                    if not left_held:
                        mouse.click(Button.left)

                if left_held:
                    mouse.release(Button.left)

            thumb_index_pinch_start = None
            left_held = False

        
        # RIGHT CLICK
        
        if middle_index_pinch and not right_clicked:
            mouse.click(Button.right)
            right_clicked = True

        elif not middle_index_pinch:
            right_clicked = False


cap.release()
cv2.destroyAllWindows()