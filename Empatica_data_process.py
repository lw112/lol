import numpy as np
import csv
import pandas as pd
import zipfile, os

class participant(object):

    def __init__(self, name, file_name):
        self.participant_name = name
        # Unzip the data file
        zip_ref = zipfile.ZipFile(file_name+'.zip', 'r')
        if not os.path.exists(self.participant_name):
            try:
                os.makedirs(self.participant_name)
                zip_ref.extractall(self.participant_name)
            except:
                raise OSError("Can't create directory (%s)!" % (self.participant_name))
        self.HR_data = []
        self.tag_data = []
        self.read_data()

    # Read the data from the csv file.
    def read_data(self):
        # Read heart rate values
        with open(self.participant_name+'/HR.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                self.HR_data.append(row[0])

        # Read time stamp log values
        with open(self.participant_name+'/tags.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                self.tag_data.append(row[0])

    # Splice the sessions according to the log values
    def get_HR_sessions(self):
        sessions = []
        start_time = float(self.HR_data[0])
        current_time = start_time
        index = 1
        new_session = []
        for time_stamps_str in self.tag_data:
            click = False
            time_stamps = float(time_stamps_str)
            while not click and index < len(self.HR_data):
                if current_time >= time_stamps:
                    sessions.append(new_session)
                    new_session = []
                    click = True
                else:
                    # Append tuple containing time elapsed since start time and avg heart rate
                    new_session.append((current_time - start_time,self.HR_data[index]))
                index += 1
                current_time += 1

        return sessions

participant1 = participant('Connor', 'Connor_data')
HR_Connor = participant1.get_HR_sessions()

participant2 = participant('Alfonso', 'Alfonso_data')
HR_Alfonso = participant2.get_HR_sessions()
HR_Alfonso

participant3 = participant('Parham', 'Parham_data')
HR_Parham = participant3.get_HR_sessions()
