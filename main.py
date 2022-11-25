# -*- coding: utf-8 -*-

import yaml
import numpy as np
import cv2

area = 12
fn_yaml = r"file_route\parking2.yml"
fn_out = r"file_route\output.avi"
config = {'save_video': False,
          'text_overlay': True,
          'parking_overlay': True,
          'parking_id_overlay': False,
          'parking_detection': True,
          'min_area_motion_contour': 60,
          'park_sec_to_wait': 5,
          'start_frame': 0}  # 35000

# Set capture device 0 for builtcam, 1 for USB cam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
if config['save_video']:
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V',
                                    'X')  # options: ('P','I','M','1'), ('D','I','V','X'), ('M','J','P','G'), ('X','V','I','D')
    out = cv2.VideoWriter(fn_out, -1, 25.0, (video_info['width'], video_info['height']))

# Read YAML data (parking space polygons)
with open(fn_yaml, 'r') as stream:
    parking_data = yaml.safe_load(stream)
parking_contours = []
parking_bounding_rects = []
parking_mask = []
for park in parking_data:
    points = np.array(park['points'])
    rect = cv2.boundingRect(points)
    points_shifted = points.copy()
    points_shifted[:, 0] = points[:, 0] - rect[0]
    # shift contour to roi
    points_shifted[:, 1] = points[:, 1] - rect[1]
    parking_contours.append(points)
    parking_bounding_rects.append(rect)
    mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1, color=255,
                            thickness=-1, lineType=cv2.LINE_8)
    mask = mask == 255
    parking_mask.append(mask)

parking_status = [False] * len(parking_data)
parking_buffer = [None] * len(parking_data)

print(
    "Program untuk mendeteksi keberadaan mobil di area parkir, tunggu 5 detik baru status berubah\nUkuran frame besar 960x720")
while (cap.isOpened()):
    spot = 0
    occupied = 0
    # Read frame-by-frame
    video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
    # video_cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) # Index of the frame to be decoded/captured next
    ret, frame = cap.read()
    if ret == False:
        print("Capture Error")
        break

    # frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    # Background Subtraction
    frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    frame_out = frame.copy()

    if config['parking_detection']:
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = parking_bounding_rects[ind]
            roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]),
                       rect[0]:(rect[0] + rect[2])]  # crop roi for faster calculation
            # print np.std(roi_gray)

            points[:, 0] = points[:, 0] - rect[0]  # shift contour to roi
            points[:, 1] = points[:, 1] - rect[1]
            # print np.std(roi_gray), np.mean(roi_gray)
            status = np.std(roi_gray) < 22 and np.mean(roi_gray) > 53
            # If detected a change in parking status, save the current time
            if status != parking_status[ind] and parking_buffer[ind] == None:
                parking_buffer[ind] = video_cur_pos

            # If status is still different than the one saved and counter is open
            elif status != parking_status[ind] and parking_buffer[ind] != None:
                if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                    print(ind + 1)
                    print(status)
                    area = ind + 1

                    parking_status[ind] = status
                    parking_buffer[ind] = None
            # If status is still same and counter is open
            elif status == parking_status[ind] and parking_buffer[ind] != None:
                # if video_cur_pos - parking_buffer[ind] > config['park_sec_to_wait']:
                parking_buffer[ind] = None
            # print(parking_status)

    if config['parking_overlay']:
        # Pengulangan untuk pewarnaan tiap kotak dari 0-11
        for ind, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[ind]:
                color = (0, 255, 0)
                spot = spot + 1
            else:
                color = (0, 0, 255)
                occupied = occupied + 1
            cv2.drawContours(frame_out, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)
            moments = cv2.moments(points)
            centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
            cv2.putText(frame_out, str(park['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_out, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        # print 'Terisi: ', occupied
        # print 'Area: ', spot

    # Draw Overlay
    if config['text_overlay']:
        # cv2.rectangle(frame_out, (1, 5), (280, 70),(255,255,255), 85)
        cv2.rectangle(frame_out, (1, 5), (300, 70), (0, 255, 0), 2)
        str_on_frame = "Parking Area Status:"
        cv2.putText(frame_out, str_on_frame, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        str_on_frame = "Empty space = %d, Occupied = %d" % (spot, occupied)
        cv2.putText(frame_out, str_on_frame, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        str_on_frame = "Last change area: " + str(area)
        cv2.putText(frame_out, str_on_frame, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # write the output frame
    # if config['save_video']:
    # if video_cur_frame % 35 == 0: # take every 30 frames
    # out.write(frame_out)

    # Display video
    imS = cv2.resize(frame_out, (960, 720))
    cv2.imshow('Parking detection - by Yaser Ali Husen', imS)
    cv2.waitKey(40)
    # cv2.imshow('background mask', bw)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        cv2.imwrite('frame%d.jpg' % video_cur_frame, frame_out)
    elif k == ord('j'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame + 1000)  # jump to frame

cap.release()
if config['save_video']: out.release()
cv2.destroyAllWindows()
