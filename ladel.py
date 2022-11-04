import argparse, os, shutil, time

import cv2
import numpy as np

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data/dataset/', help='dataset path')
parser.add_argument('--video', type=str, default='./data/input_video.mp4', help='input video path')
parser.add_argument('--labelmap', type=str, default='./data/labelmap.txt', help='labelmap path')
parser.add_argument('--img-size', nargs='+', type=int, default=640, help='train image sizes, always a square')
parser.add_argument('--tracker', type=str, default='csrt', help='name of opencv tracker for labeling')
parser.add_argument('--mode', type=str, default='tflite', help='labeling mode')

opt = parser.parse_args()

labels = []
trackers = cv2.legacy.MultiTracker_create()

cap = cv2.VideoCapture(opt.video)

is_dataset_paths = os.path.exists(opt.dataset + "images/") or os.path.exists(opt.dataset + "labels/")

do_purge = False
if is_dataset_paths:
    do_purge = input("Purge? (y/N) > ").lower() == 'y'

    if do_purge:
        for i in range(5, 0, -1):
            print("Purging dataset in... " + str(i), end="\r")
            time.sleep(1)

        shutil.rmtree(opt.dataset + "images/")
        shutil.rmtree(opt.dataset + "labels/")
        print("PURGED")

is_dataset_paths = os.path.exists(opt.dataset + "images/") or os.path.exists(opt.dataset + "labels/")
if not is_dataset_paths:
    os.mkdir(opt.dataset + "images/")
    os.mkdir(opt.dataset + "labels/")

if opt.mode == 'tflite':
    csv_writer = open(opt.dataset + "labels/annotations.csv", 'w+')
    csv_writer.write("filename,xmin,xmax,ymin,ymax,class\n")

def contains(pos, bb):
    return pos[0]>bb[0] and pos[0]<bb[0]+bb[2] and pos[1]>bb[1] and pos[1]<bb[1]+bb[3]

def onMouse(event, x, y, flags, param):
    global labels
    global trackers

    if event == cv2.EVENT_LBUTTONDOWN:
        tracker_arr = list(trackers.getObjects())
        for bb in tracker_arr:
            if contains((x, y), bb):
                trackers.clear()
                trackers = cv2.legacy.MultiTracker_create()

                idx = tracker_arr.index(bb) # remove clicked tracker from new array
                tracker_arr.pop(idx) # remove from arr
                labels.pop(idx) # remove from labels

                for bb in tracker_arr: # add back the rest of the trackers
                    tracker = OPENCV_OBJECT_TRACKERS[opt.tracker]()
                    trackers.add(tracker, frame, bb)

def select_objs(frame, trackers):
    global labels

    box = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)
    box = tuple(map(tuple, box))

    for bb in box:
        tracker = OPENCV_OBJECT_TRACKERS[opt.tracker]()
        trackers.add(tracker, frame, bb)
        if opt.mode == 'tflite':
            print("cannot be an int :(")
        labels.append(input("Enter label > "))

    cv2.setMouseCallback("Frame", onMouse)

    return labels

def update(frame):
    global frame_num
    global play

    if not np.any(trackers.getObjects()):
        select_objs(frame, trackers)
        first_frame = False
        play = False

    frame_name = str(frame_num) + ".jpg"
    cv2.imwrite(opt.dataset + 'images/' + frame_name, frame)

    (success, boxes) = trackers.update(frame)

	# loop over the bounding boxes and draw them on the frame/write them to the csv
    for box_n in range(len(boxes)):
        (x, y, w, h) = [int(v) for v in boxes[box_n]]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if opt.mode == 'yolov7':
            open(opt.dataset + 'labels/' + str(frame_num) + ".txt", 'w').write(
                f"{labels[box_n]} {(x+(0.5*w)) / opt.img_size} {(y+(0.5*h)) / opt.img_size} {(w) / opt.img_size} {(h) / opt.img_size}\n"
            )
        elif opt.mode == 'tflite':
            csv_writer.write(
                # an actually sane data format
                f"{str(frame_num) + '.jpg'},{x},{x + w},{y},{y + h},{labels[box_n]}\n"
            )
    cv2.imshow("Frame", frame)

frame_num = max([int(name.split('.')[0]) for name in os.listdir(opt.dataset + 'images/')]) if not do_purge and input("Use last (Y/n) > ").lower() =='y' else 0
play = True
while cap.isOpened():
    key = cv2.waitKey(1) & 0xFF

    if play or key == ord('d'):
        frame_num += 1
        # if frame_num > 0: cap.set(2, (frame_num /(cap_duration*cap_fps)))
        if os.path.exists(opt.dataset + 'images/' + str(frame_num) + '.jpg'):
            print("Gunna go back in time 🎶")
            frame = cv2.imread(opt.dataset + 'images/' + str(frame_num) + '.jpg')
        else:
            ret, frame = cap.read()
            if not ret: break
        frame = cv2.resize(frame, (opt.img_size, opt.img_size))
        update(frame)
    elif key == ord('a'):
        frame_num -= 1
        frame = cv2.imread(opt.dataset + 'images/' + str(frame_num) + '.jpg')
        frame = cv2.resize(frame, (opt.img_size, opt.img_size))
        update(frame)
    elif key == ord('x'):
        frame_num += 1
        ret, frame = cap.read()
        update(frame)

    if key == ord('i'): # add bounding boxes
        labels = select_objs(frame, trackers)
    elif key == ord('s'): # start over
        trackers.clear()
        trackers = cv2.legacy.MultiTracker_create()

        labels = []
        labels = select_objs(frame, trackers)
    elif key == ord('p'):
        play = not play
    elif key == ord('q'):
        break

csv_writer.close()

cap.release()
cv2.destroyAllWindows()
