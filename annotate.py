import os
import sys
from datetime import datetime as dt
from glob import glob

from tqdm import tqdm
import numpy as np
import cv2
import dlib

from session import Session
from tools import done, shape_to_np

predictor_path = 'face_tracker/shape_predictor_68_face_landmarks.dat'

face_detector = dlib.get_frontal_face_detector()
points_fitter = dlib.shape_predictor(predictor_path)


def fit_face_points(image):
    face_boxes, face_points = face_detector(image, 1), []

    if len(face_boxes):
        face_box = face_boxes[0]
        face_points = shape_to_np(points_fitter(image, face_box))

    return face_boxes, face_points

def main():
    # either pass in a list of indices (space separated) to load
    # OR pass in one index, which is the 'starting point'
   
    if len(sys.argv) > 1:
        indices = sorted(map(int, sys.argv[1:]))
        all_sessions_paths = ['data/Sessions/' + str(idx) for idx in indices]
    elif len(sys.argv) == 2:
    	indices = sorted(map(int, sys.argv[1:]))

    else:
        all_sessions_paths = sorted(glob('data/Sessions/*'))
        #all_sessions_paths = ['data/Sessions/1']

    all_sessions = {}

    t1 = dt.now()
    print(t1, 'Loading data')

    total_number_of_frames = 0
    fails = []

    for path in all_sessions_paths:
        try:
            index = path.split('/')[-1]
            session = Session(index)

            if not session.laughs_extracted:
            	session.extract_laughter_subclips_from_video()



        except Exception as e:
            print(e)
            fails.append(index)


    t2 = dt.now()
    print(t2, 'Loading data took', t2-t1)
    print('Total number of frames', total_number_of_frames)
    print('Failed extracting for sessions', fails)

    done()

if __name__ == '__main__':
    main()