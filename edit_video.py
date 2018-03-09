import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


m = '/Users/laura/Dropbox/dev/lol/data/Sessions/1/S001-001.avi'
a = '/Users/laura/Dropbox/dev/lol/data/Sessions/1/S001-001_mic.wav'

video = mp.VideoFileClip(m).subclip(35.00, 38.00)
audio = mp.AudioFileClip(a).subclip(35.00, 38.00)
video = video.set_audio(audio)

#final = ffmpeg_extract_subclip(video, 35.00, 38.00)

#help(video.write_videofile)

video.write_videofile("output.mp4", codec='libx264')
