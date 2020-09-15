import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob
import os
import copy
import scipy.io
import math
from pdb import set_trace
from itertools import combinations
from itertools import product

from detect_hands import detect_hands

import os, sys, inspect, linecache
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

#Body object, estimation
from src import model
from src import util
from src.body_hook import Body

# Returns two dictionaries storing 2d numpy arrays containing pose data for each video in a video set

class VidHooking():

    def __init__(self, dir):
            
            self.body_hooking = Body('../model/body_pose_model.pth')
            self.body_estimation = Body('../model/body_pose_model.pth', hooking=False)
            self.vid_list = self.get_vid_data(dir)
            
            set_trace()
            
            self.full_features = self.get_full_features()
    
    def get_vid_data(self, dir):
        
        vid_list = []
        
        # Opens and stores all video
        for filename in glob.glob(dir+'*.mp4'):

            cap = cv2.VideoCapture(filename)
            
            while not cap.isOpened(): # The video has not loaded yet. Gives more time to load
                cap = cv2.VideoCapture(filename)
                cv2.waitKey(1000)
                print('Wait for the header')
                
            vid_list.append([cap, filename.split('/')[2][:-4]])
        
        vid_list.sort(key = lambda x: int(x[1].split('vid')[1].split('.')[0])) # Sorts numerically
        
        return vid_list
    
    def get_full_features(self):
    
        full_features = {}
        for i in range(0,56):
            full_features[i] = {}
        
        # Cycles through each video in vid_list
        for counter, vid_info in enumerate(self.vid_list, 1): # vid_info[video, filename]
            
            try:
            
                cap = vid_info[0]
                vid_name = vid_info[1]
                
                for layer in full_features:
                    full_features[layer][vid_name] = []
                
                tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Selects frames to skip so to analyze exactly 75 frames
                # 75 = lowest number of total frames in set1; 78 = lowest number of total frames in set2
                if not tot_frame == 75:
                   skip_mult = tot_frame/(tot_frame - 75) # Because videos have different frame lengths, this variable is used to skip over multiples of its value so that after having read through the video in its entirety, only 75 frames have been analyzed.
                   skip_val = [round(skip_mult*(i+1)) for i in range(tot_frame-75)] # Stores frame indexes to be skipped
                else:
                   skip_val = []
                
                frames_read = 0
                while frames_read < 75:
                
                    flag, frame = cap.read() # read in the next frame
                    pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # frame index

                    if not flag: # The next frame is not ready, so we try to read it again
                       cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                       cv2.waitKey(1000)

                    elif pos_frame not in skip_val: # Frame number is not skipped
                        frames_read += 1
                        features = self.body_hooking(frame)
                        candidate, subset = self.body_estimation(frame)
                        features[55] = self.get_final_pose(candidate, subset, frame.shape[0])
                        for layer in full_features:
                            tensor = features[layer]
                            full_features[layer][vid_name].append(tensor)
            
            except Exception as e:
                cap.release()
                tb = sys.exc_info()[-1]
                f = tb.tb_frame; l = tb.tb_lineno
                filename = f.f_code.co_filename
                print('{}: An exception occurred in {} at line {} on frame {}. Error: {}'.format(vid_info[1], filename, l, pos_frame, repr(e)))
                    
        return full_features
    
    def get_final_pose(self, candidate, subset, frame):
        set_trace()
        dim = int(frame/2)
        im = [[(-1,-1) for j in range(dim)] for i in range(dim)]
        for i, person in enumerate(subset):
            for j, joint in enumerate(person):
                if int(joint) != -1:
                    x = int(candidate[int(joint)][0]/2)
                    y = int(candidate[int(joint)][1]/2)
                    im[x][y] = (i, j)
        return np.array(im).flatten()
                            
if __name__ == "__main__":
    
    dir = 'exp/demo/'
    hooking = VidHooking(dir)
    set_trace()
