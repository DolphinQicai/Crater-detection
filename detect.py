import threading
from ultralytics import YOLO
from PIL import Image
import cv2
import os
from tqdm import trange
import config1280x480
import numpy as np
import func
import queue

model = YOLO("best.pt")
path = 'http://192.168.137.236:8080/?action=stream'
cap = cv2.VideoCapture(path)
distance = 10000

frame_queue = queue.Queue()
result_queue = queue.Queue()
distance_queue = queue.Queue()

def detect_frames():
    while True:
        frame = frame_queue.get()[0:480, 0:640]
        mask = frame

        results = model(frame, max_det=5, conf=0.8)
        boxes = results[0].boxes
        area = 0
        index = 0
        for idx in range(len(boxes)):
            box = boxes[idx]
            area_box = float(box.xywh[0][2]) * float(box.xywh[0][3])
            if area_box > area:
                area = area_box
                index = idx
        try:
            xy = boxes[index].xyxy[0]
            start = (int(xy[0]), int(xy[1]))
            end = (int(xy[2]), int(xy[3]))
            color = (0, 0, 255)
            thickness = 1
            annotated_frame = results[0].orig_img
            center = [(start[0]+end[0]) / 2, (start[1]+end[1]) / 2]
            if center[0] > 370 and center[1] > 290:
                text = 'right down'
            elif center[0] < 270 and center[1] > 290:
                text = 'left down'
            elif center[0] > 370 and center[1] < 190:
                text = 'right up'
            elif center[0] < 270 and center[1] < 190:
                text = 'left up'
            
            elif center[0] < 270 and center[1] > 190 and center[1] < 290:
                text = 'left'
            elif center[0] > 370 and center[1] > 190 and center[1] < 290:
                text = 'right'
            elif center[0] > 270 and center[0] < 370 and center[1] < 190:
                text = 'up'
            elif center[0] > 270 and center[0] < 370 and center[1] > 290:
                text = 'down'
            else:
                text = 'OK'
                
            try:
                dis = distance_queue.get()
            except:
                dis = 10000
            
            text2 = 'dis: ' + '%0.3f'%(dis) + 'm'
            mask = cv2.rectangle(annotated_frame, start, end, color, thickness, )
            if text == 'OK' and dis < 0.2:
                text3 = 'Success!'
            else:
                text3 = ''
            cv2.putText(mask, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(mask, text2, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(mask, text3, (10, 120), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
        except:
            pass

        result_queue.put(mask)

def get_distance():
    while True:
        try:
            single_frame = list(frame_queue.queue)[0]
            iml = single_frame[0:480, 0:640]
            imr = single_frame[0:480, 640:1280]
            distance = func.get_distance(iml, imr)
            distance_queue.put(distance)
        except:
            pass

def show_result():
    while True:
        mask = result_queue.get()
        cv2.imshow("YOLOv8 Inference", mask)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
def capture():
    while True:
        success, frame_all = cap.read()
        frame_queue.put(frame_all)

def main():
    capture_thread = threading.Thread(target=capture)
    crater_detect_thread = threading.Thread(target=detect_frames)
    distance_measurement_thread  = threading.Thread(target=get_distance)
    show_result_thread = threading.Thread(target=show_result)

    capture_thread.start()
    crater_detect_thread.start()
    distance_measurement_thread.start()
    show_result_thread.start()

          

    capture_thread.join()
    crater_detect_thread.join()
    distance_measurement_thread.join()
    show_result_thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()