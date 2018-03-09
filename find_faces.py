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
    
    if len(sys.argv) > 1:
        indices = sys.argv[1:]
        print(indices)
        all_sessions_paths = ['data/Sessions/' + idx for idx in indices]
    else:
        all_sessions_paths = sorted(glob('data/Sessions/*'))
        #all_sessions_paths = ['data/Sessions/1']

    all_sessions = {}

    t1 = dt.now()
    print(t1, 'Loading data')

    fails = []

    for path in all_sessions_paths:
        try:
            index = path.split('/')[-1]
            session = Session(index)

            session.extract_laughter_subclips_from_video()
            #session.read_audio_file()

            # extract all frames containing laughter from video
            #if not session.frames_extracted:
            #    session.extract_frames_from_video()

            #cmd = 'ffmpeg -i S001-001.avi -i S001-001_mic.wav -vcodec copy -acodec copy muxed_file.avi'

            #session.read_frames()
            #print ('session %d: %d frames' % index, len(session.all_frames_paths))


            # find all face points for each extracted image
            #if not session.points_extracted:
            #    for idx, frame in session.all_frames.items():
            #        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #        face_boxes, face_points = fit_face_points(gray)
            #        session.save_points(idx, face_points)
        except Exception as e:
            print(e)
            fails.append(index)


    t2 = dt.now()
    print(t2, 'Loading data took', t2-t1)
    #print('Total number of frames', total_number_of_frames)
    #print('Failed extracting for sessions', fails)

    done()

if __name__ == '__main__':
    main()