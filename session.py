from glob import glob
import os

import pandas as pd
import numpy as np
from scipy.io import wavfile
import cv2
import av
from datetime import datetime as dt
import moviepy.editor as mp

from tools import frame_path_to_idx

sessions_dir = 'data/Sessions/'
frame_format = '.jpg'
point_format = '.txt'

categories = ['Laughter', 'SpeechLaughter', 'PosedLaughter']

class Session(object):
    def __init__(self, idx):
        self.idx = int(idx)

        self.session_path = sessions_dir + str(idx) + '/'
        self.frames_path = self.session_path + 'frames/'
        self.points_path = self.session_path + 'points01/'

        self.csv_file = self.session_path + 'laughterAnnotation.csv'
        self.csv = pd.read_csv(self.csv_file)
        
        if not len(self.csv):
            raise Exception('No LOLs found for session %d. (empty csv)' % self.idx)

        self.video_file = glob(self.session_path + '/S???-???.avi')[0]
        self.laugh_subclips = sorted(glob(self.session_path + '/S???-???-l???.avi'))
        self.audio_file = glob(self.session_path + '/S*_mic.wav')[0]
        self.audio_rate = 0
        self.audio = dict()

        self.length = 0.0

        self.csv_times = dict()
        self.csv_frames_indices = dict()

        self.setup()

        self.all_frames_paths = sorted(glob(self.frames_path + '*' + frame_format))
        self.all_points_paths = sorted(glob(self.points_path + '*' + point_format))

        self.frames_extracted = bool(len(self.all_frames_paths))
        self.points_extracted = bool(len(self.all_points_paths))
        self.laughs_extracted = bool(len(self.laugh_subclips))

        self.all_frames = dict()
        self.all_points = dict()

    def setup(self):
        if not os.path.isdir(self.frames_path):
            os.mkdir(self.frames_path)

        if not os.path.isdir(self.points_path):
            os.mkdir(self.points_path)

        if os.path.isfile(self.csv_file):
            self.csv = pd.read_csv(self.csv_file)

        if len(self.csv):
            for idx, row in self.csv.iterrows():
                if row['Type'] in categories:
                    self.csv_times[idx] = (row["Start Time (sec)"], row["End Time (sec)"])
                    self.csv_frames_indices[idx] = list(range(row['Start Frame'], row['End Frame']))

            print(dt.now(), 'session', self.idx, 'found', len(self.csv_times), 'laughs')

    def extract_laughter_subclips_from_video(self):
        avi, wav = self.video_file, self.audio_file

        for idx, (start, end) in self.csv_times.items():
            out = avi.replace('.avi', '-l' + str(idx).zfill(3) + '.mp4')
            
            v = mp.VideoFileClip(self.video_file).subclip(start, end)
            a = mp.AudioFileClip(self.audio_file).subclip(start, end)
            v = v.set_audio(a)
            v.write_videofile(out, codec='libx264')

        self.laugh_subclips = sorted(glob(self.session_path + '/S???-???-l???.avi'))
        self.laughs_extracted = bool(len(self.laugh_subclips))

        print(dt.now(), 'session', self.idx, 'extracted', len(self.laugh_subclips), 'subclips')

    def extract_frames_from_video(self):
        container = av.open(self.video_file)
        frames_to_extract = []
        for idx, frames in self.csv_frames_indices.items():
            frames_to_extract += frames

        for idx, frame in enumerate(container.decode(video=0)):
            if idx in frames_to_extract:
                frame.to_image().save(self.frames_path + 'frame-%04d.jpg' % frame.index)

        self.all_frames_paths = sorted(glob(self.frames_path + '*.jpg'))
        self.frames_extracted = bool(len(self.all_frames_paths))

        print(dt.now(), 'session', self.idx, 'extracted', len(self.all_frames_paths), 'frames')

    def read_audio_file(self):
        rate, signal = wavfile.read(self.audio_file)
        self.audio_rate = rate
        self.length = len(signal)/rate

        if len(self.csv):
            for index, (start, end) in self.csv_times.items():
                start_t, end_t = int(start*rate), int(end*rate)
                self.audio[index] = signal[start_t: end_t]
        else:
            self.audio = {0: signal}

    def read_frames(self):
        self.all_frames_paths = sorted(glob(self.frames_path + '*' + frame_format))
        for frame_path in self.all_frames_paths:
            idx = frame_path_to_idx(frame_path)
            self.all_frames[idx] = cv2.imread(frame_path)
        self.frames_extracted = bool(len(self.all_frames_paths))

    def read_points(self):
        self.all_points_paths = sorted(glob(self.points_path + '*' + point_format))
        for points_file in self.all_points_paths:
            self.all_points.append(np.loadtxt(points_file, dtype=int))
        self.points_extracted = bool(len(self.all_points_paths))

    def save_points(self, idx, points):
        path = self.points_path + 'points-%04d' % idx + point_format
        np.savetxt(path, points, fmt='%d')
