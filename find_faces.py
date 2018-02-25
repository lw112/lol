import os
from datetime import datetime as dt
from glob import glob

from tqdm import tqdm
import numpy as np
import cv2
import dlib

from session import Session
from tools import done, shape_to_np

predictor_path = 'data/classifiers/shape_predictor_68_face_landmarks.dat'

face_detector = dlib.get_frontal_face_detector()
points_fitter = dlib.shape_predictor(predictor_path)


def fit_face_points(image):
    face_boxes, face_points = face_detector(image, 1), []

    if len(face_boxes):
        face_box = face_boxes[0]
        face_points = shape_to_np(points_fitter(image, face_box))

    return face_boxes, face_points

all_sessions = {}
all_sessions_paths = sorted(glob('data/Sessions/*'))

t1 = dt.now()
print(t1, 'Loading data')

total_number_of_frames = 0
for path in tqdm(all_sessions_paths):
    index = path.split('/')[-1]
    session = Session(index, True)
    session.read_audio_file()

    if not session.frames_extracted:
        session.extract_frames_from_video()

    session.read_frames()

    total_number_of_frames += len(session.all_frames_paths)

    if not session.points_extracted:
        for idx, frame in session.all_frames.items():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_boxes, face_points = fit_face_points(gray)
            session.save_points(idx, face_points)

t2 = dt.now()
print(t2, 'Loading data took', t2-t1)

print('Total number of frames', total_number_of_frames)
done()