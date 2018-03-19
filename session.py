from glob import glob
import os

import pandas as pd
import numpy as np
from scipy.io import wavfile
import cv2
import av
from datetime import datetime as dt
import moviepy.editor as mp
import h5py

from tools import frame_path_to_idx

sessions_dir = 'data/Sessions/'
frame_format = '.jpg'
point_format = '.txt'
subclip_format = '.mp4'

categories = ['Laughter', 'PosedLaughter']


class Session(object):
    def __init__(self, idx):
        self.idx = int(idx)
        self.idx_str = str(idx).zfill(3)

        self.session_path = sessions_dir + str(idx) + '/'
        self.frames_path = self.session_path + 'frames/'
        self.points_path = self.session_path + 'points01/'
        self.image_features_path = self.session_path + 'features_images_' + self.idx_str + '.h5'

        self.csv_file = self.session_path + 'laughterAnnotation.csv'
        self.csv = pd.read_csv(self.csv_file)
        
        if not len(self.csv):
            raise Exception('No LOLs found for session %d. (empty csv)' % self.idx)

        self.video_files = glob(self.session_path + '/S???-???.avi')
        self.video_file = self.video_files[0] if len(self.video_files) else glob(self.session_path + '/S*.avi')[0]
        self.laughter_subclips = sorted(glob(self.session_path + '/S???-???-l???' + subclip_format))
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
        self.laughs_extracted = bool(len(self.laughter_subclips))

        self.all_frames = dict()
        self.all_points = dict()

        self.all_laughts_labels = dict()

    def setup(self):
        if not os.path.isdir(self.frames_path):
            os.mkdir(self.frames_path)

        #if not os.path.isdir(self.points_path):
        #    os.mkdir(self.points_path)

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
            out = avi.replace('.avi', '-l' + str(idx).zfill(3) + subclip_format)
            
            v = mp.VideoFileClip(self.video_file).subclip(start, end)
            a = mp.AudioFileClip(self.audio_file).subclip(start, end)
            v = v.set_audio(a)
            v.write_videofile(out, codec='libx264')

        self.laughter_subclips = sorted(glob(self.session_path + '/S???-???-l???' + subclip_format))
        self.laughs_extracted = bool(len(self.laughter_subclips))

        print(dt.now(), 'session', self.idx, 'extracted', len(self.laughter_subclips), 'subclips')

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
            idx = frame_path_to_idx(points_file)
            self.all_points[idx] = np.loadtxt(points_file, dtype=int)
        self.points_extracted = bool(len(self.all_points_paths))

    def save_points(self, idx, points):
        path = self.points_path + 'points-%04d' % idx + point_format
        np.savetxt(path, points, fmt='%d')

    def save_video_features(self, features):
        with h5py.File(self.image_features_path, 'w') as file:

            for key, data in features.items():
                if key == 'laughs':
                    f1 = file.create_group('laughs')
                    for kk, indices in features[key].items():
                        f1.create_dataset(name=str(kk), data=indices)

                else:
                    file.create_dataset(name=key, data=data)

    def read_video_features(self):
        features = dict()
        with h5py.File(self.image_features_path, 'r') as file:
            labels = list(file.keys())
            for label in labels:
                if label == 'laughs':
                    features[label] = dict()
                    for k, v in file['laughs'].items():
                        features[label][k] = list(v)
                else:
                    features[label] = np.array(file.get(label))
        return features

    def idx_to_frame_path(self, idx):
        return self.frames_path + 'frame-' + str(idx).zfill(4) + '.jpg'