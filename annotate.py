import os
import sys
from datetime import datetime as dt
from glob import glob

from tqdm import tqdm
import numpy as np
import cv2
import dlib

from session import Session
from tools import done

def main():

    if sys.argv[2] == '-':
        indices = range(int(sys.argv[1]), nb_sessions)
    elif len(sys.argv) > 1:
        indices = sys.argv[1:]
    else:
        indices = range(nb_sessions)


    all_sessions = {}

    t1 = dt.now()
    print(t1, 'Loading data')


    for idx in indices:
        try:
            session = Session(index)

            if not session.laughs_extracted:
            	session.extract_laughter_subclips_from_video()

            #TO-DO: implement this
            #for subclip in session.laughter_subclips:
            	#play videos
            	


            all_sessions[idx] = session

        except Exception as e:
            print(e)
            fails.append(index)


    t2 = dt.now()
    print(t2, 'Loading data took', t2-t1)
    print('Total number of frames', total_number_of_frames)
    print('Failed extracting for sessions', fails)

    done()

if __name__ == '__main__':
    # command line args
    # python annotate.py [session_no]       --> annotate this session only
    # python annotate.py [session_no] -     --> annotate this session and all following
    # python annotate.py                    --> annotate all sesssion starting at 1

    main()