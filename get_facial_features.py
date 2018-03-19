import os
import sys
from datetime import datetime as dt
from glob import glob
from collections import defaultdict
import json

from tqdm import tqdm
import numpy as np
import cv2
import dlib

from session import Session
from tools import done, shape_to_np


face_finder = cv2.CascadeClassifier("face_tracker/haarcascade_frontalface_default.xml")

predictor_path = 'face_tracker/shape_predictor_68_face_landmarks.dat'

face_detector = dlib.get_frontal_face_detector()
points_fitter = dlib.shape_predictor(predictor_path)

nb_sessions = 191
resized_shape = 100


def main():

    if len(sys.argv) == 2:
        indices = sys.argv[1:]
    elif len(sys.argv) == 3 and sys.argv[2] == '-':
        indices = range(int(sys.argv[1]), nb_sessions)      
    else:
        indices = range(1, nb_sessions)

    indices = [59]
    t1 = dt.now()
    #print(t1, 'Loading sessions', indices)

    all_sessions, fails = {}, []
    for index in indices:
        try:
            session = Session(index)
        except:
            continue

        # extract all frames containing laughter from video
        #if not session.frames_extracted:
         #   session.extract_frames_from_video()

        cropped_images, cropped_points, found_faces, all_labels = [], [], [], []

        # find all face points for each extracted image
        for idx, frames in tqdm(session.csv_frames_indices.items()):
            for frame_idx in frames:
                try:
                    frame_path = session.idx_to_frame_path(frame_idx)
                    if os.path.isfile(frame_path):
                        frame = cv2.imread(frame_path)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        boxes = find_face_boxes(gray)

                        if not len(boxes):
                            continue

                        box = boxes[0]
                        cropped = crop_image_to_face(gray, box)
                        cropped_images.append(cropped)
                        all_labels.append(frame_idx)

                        boxes, points = fit_face_points(cropped, face_detector, points_fitter)

                        if len(points):
                            clipped_points = np.clip(points, 0, resized_shape - 1)
                            cropped_points.append(clipped_points)
                            found_faces.append(frame_idx)

                except Exception as e:
                    print('ERROR', idx, frame_idx)
                    print(e)
                    fails.append(index)

        features = {'all_labels': np.array(all_labels), 'images': np.array(cropped_images), 'points': np.array(cropped_points),
                    'faces_found': np.array(found_faces), 'laughs': session.csv_frames_indices}

        session.save_video_features(features)

        #_features = session.read_video_features()

    t2 = dt.now()
    print(t2, 'Getting data took', t2-t1)
    #print('Total number of frames', total_number_of_frames)
    #print('Failed extracting for sessions', fails)

    done()


def find_face_boxes(gray):
    return face_finder.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)


def crop_image_to_face(gray, box):

    pad = 20
    x, y, w, h = box
    cropped = gray[y - pad:y + h + pad, x - pad:x + w + pad]
    resized = cv2.resize(cropped, (resized_shape, resized_shape))

    return resized


def fit_face_points(img, face_detector, points_fitter):
    face_boxes, face_points = face_detector(img, 1), []

    if len(face_boxes):
        face_box = face_boxes[0]
        face_points = shape_to_np(points_fitter(img, face_box))

    return face_boxes, face_points


if __name__ == '__main__':
    # command line args
    # python get_facial_features.py [session_no]       --> get data for this session only
    # python get_facial_features.py [session_no] -     --> get data for this session and all following
    # python get_facial_features.py                    --> get data for all sesssion starting at 1
    main()