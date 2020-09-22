import cv2
import numpy as np
import glob
from scipy.io import savemat
from pdb import set_trace

from scipy.stats import zscore
from scipy.spatial.distance import pdist

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
        self.vid_list = self.read_vid_data(dir)
#        self.frame_list = self.get_frame_info()
    
    def read_vid_data(self, dir):
        
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
        rdms = {} # Eventually store 56 rdms
        
        # Cycles through each hooking layer
        for layer in range(0,56):
            
            avgd_data_for_layer = []
            
            # Cycles through each video in vid_list
            for counter, vid_info in enumerate(self.vid_list, 1): # vid_info[video, filename]

                data = self.get_vid_data_for_layer(vid_info, layer)
                avgd_data_for_layer.append(data)
        
            distance_vector = pdist(avgd_data_for_layer, metric='sqeuclidean')
            distance_vector = zscore(distance_vector)
            rdms['Layer_{}'.format(layer)] = distance_vector
                    
        return rdms
    
    def get_vid_data_for_layer(self, vid_info, layer):

        '''
        Cycles through each frame in a video, summing together all output arrays at
        layer=layer. Returns the averaged array.
        '''

        try:

            cap = vid_info[0]
            vid_name = vid_info[1]
            
            return_data = None
            
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
                    
                    # first frame, sets the shape for the proceeding array additions
                    if frames_read == 0:
                        if layer < 55:
                            return_data = np.array(self.body_hooking(frame, layer_to_hook=layer))
                        else:
                            candidate, subset = self.body_estimation(frame)
                            return_data = np.array(self.get_final_pose(candidate, subset, frame.shape[0]))
                    else:
                        if layer < 55:
                            features = np.array(self.body_hooking(frame, layer_to_hook=layer))
                        else:
                            candidate, subset = self.body_estimation(frame)
                            features = np.array(self.get_final_pose(candidate, subset, frame.shape[0]))
                        return_data = np.add(return_data, features)
            
                frames_read += 1
        
        except Exception as e:
            cap.release()
            tb = sys.exc_info()[-1]
            f = tb.tb_frame; l = tb.tb_lineno
            filename = f.f_code.co_filename
            print('{}: An exception occurred in {} at line {} on frame {} for hooking layer {}. Error: {}'.format(vid_info[1], filename, l, pos_frame, layer, repr(e)))
        
        print('Layer {}: {} done'.format(layer, vid_name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #cap.release()
        
        return return_data/frames_read
        
    def get_final_pose(self, candidate, subset, frame):
    
        '''
        Retrieves final pose information. Creates a 2d-list of the same size as the
        frame. If a joint is detected at a certain point, the element reflects the
        joint type and assigned person. Returns a flattened version of list.
        '''
    
        dim = int(frame/2)
        im = [[(-1,-1) for j in range(dim)] for i in range(dim)]
        for i, person in enumerate(subset):
            for j in range(18):
                joint = int(person[j])
                if joint != -1:
                    x = int(candidate[joint][0]/2)
                    y = int(candidate[joint][1]/2)
                    im[x][y] = (i, j)
                    
        return np.array(im).flatten()
                            
if __name__ == "__main__":
    
    dir = 'exp/vids_set/'
    hooking = VidHooking(dir)
    rdms = hooking.get_full_features()
    savemat(dir+dir.split('/')[1]+'_hooking_RDMs.mat', rdms)
    set_trace()
