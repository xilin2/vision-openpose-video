'''
Code segments from: https://stackoverflow.com/questions/18954889/how-to-process-images-of-a-video-frame-by-frame-in-video-streaming-using-openc
'''

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

from detect_hands import detect_hands

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

#Body object, estimation
from src import model
from src import util
from src.body import Body

#Hand object, estimation
from src.hand_vid import Hand

# clean up; Ensures that the boundaries of hand box does not exceed frame
def bound_to_frame(x, y, w, h, frame):

    height, width = len(frame[0]), len(frame[1])
    
    if x+w > width:
        w = width-x
    if x < 0:
        x = 0
    elif x > width:
        x = width
        
    if y+h > height:
        h = height-y
    if y < 0:
        y = 0
    elif y > height:
        y = height
        
    return x, y, w, h
        

# Returns image file associated with MPL plot
def get_img_from_fig(plt):
    plt.savefig(dir+'temp.png')
    im = cv2.imread(dir+'temp.png')
    #os.remove(dir+'temp.png')
    return im

# Returns features to be added to dataset
def get_features(subset, candidate):
    x = []
    subset = subset.flatten()
    for i in range(18): # cycle through each body part
        if int(subset[i]) == -1:
            x.extend([-1,-1,-1,-1])
        else:
            x.extend(candidate[int(subset[i])])
    return x

# Returns coordinate data only
def get_coords(subset, candidate):
    x = []
    subset = subset.flatten()
    for i in range(18):
        if int(subset[i]) == -1:
            x.append([])
        else:
            x.append(candidate[int(subset[i])][0:2])
    return x

# Finds average distance between keypoints for two subsets
def get_average_dist(prev, curr):
    num, tot = 0, 0
    for i in range(18):
        if all(p == 0 for p in prev[i]) or all(c == 0 for c in curr[i]):
        #if not prev[i] or not curr[i]:
            continue
        else:
            px, py, cx, cy = prev[i][0], prev[i][1], curr[i][0], curr[i][1]
            x, y = px - cx, py - cy
            tot += math.sqrt(x**2 + y**2)
            num += 1
    if num == 0:
        return 1000
    else:
        return tot/num
            
# Returns two dictionaries storing 2d numpy arrays containing pose data for each video in a directory
def get_vid_data(dir):

    vid_list = []
    body_list = {}
    hand_list = {}
    body_estimation = Body('../model/body_pose_model.pth')
    hand_estimation = Hand('../model/hand_pose_model.pth')

    if not os.path.exists(dir+'op'):
        os.makedirs(dir+'op')

    # Opens and stores all vidoes
    for filename in glob.glob(dir+'*.mp4'):

        cap = cv2.VideoCapture(filename)
        
        while not cap.isOpened(): # The video has not loaded yet. Gives more time to load
            cap = cv2.VideoCapture(filename)
            cv2.waitKey(1000)
            print('Wait for the header')
            
        vid_list.append([cap, filename.split('/')[2][:-4]])
    
    vid_list.sort(key = lambda x: int(x[1].split('vid')[1].split('.')[0])) # Sorts numerically

    # Cycles through each video in vid_list
    for counter, vid_info in enumerate(vid_list, 1): # vid_info[video, filename]
    
        try:
            cap = vid_info[0]
            tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Selects frames to skip so to analyze exactly 75 frames
            # 75 = lowest number of total frames in set1
            # 78 = lowest number of total frames in set2
            if not tot_frame == 75:
                skip_mult = tot_frame/(tot_frame - 75) # Because videos have different frame lengths, this variable is used to skip over multiples of its value so that after having read through the video in its entirety, only 75 frames have been analyzed.
                skip_val = [round(skip_mult*(i+1)) for i in range(tot_frame-75)] # Stores frame indexes to be skipped
            else:
                skip_val = []
            
            length = tot_frame / cap.get(cv2.CAP_PROP_FPS)
            fps = 75/length

            b_features = [] # 2d array to temporarily store each video's pose features, dim = [frame count=75), 18*4=72]
            h_features = [] # dim = [75, 2*21*2=126]
            frames = [] # stores drawn frames to later write into video

            prev_body = [] # stores the last detected body pose
            
            while True:
                
                flag, frame = cap.read() # read in the next frame
                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # frame index
                
                if not flag: # The next frame is not ready, so we try to read it again
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                    cv2.waitKey(1000)
                
                elif pos_frame not in skip_val: # Frame number is not skipped
                    
                    ''' Body '''
                    
                    candidate, subset = body_estimation(frame)
                    
                    if len(subset) < 1: # No person detected. Creates an empty features array and resets previous body to []
                        while len(subset) != 1:
                            subset = np.append(subset, [np.ones(20)*-1], axis=0)
                        prev_body = [] # resets prev_body
                    
                    elif len(prev_body) == 0: # No previous body stored, more than 1 person detected. Sets person with most keypoints as new body.
                        if len(subset) > 1: # Reduces subset to person with most keypoints
                            body_parts_count = []
                            for person in subset:
                                body_parts_count.append(person[19])
                            while len(subset) != 1:
                                smallest = body_parts_count.index(min(body_parts_count))
                                subset = np.delete(subset, smallest, axis=0)
                                body_parts_count.pop(smallest)
                        p = get_coords(subset, candidate)
                        prev_body = copy.deepcopy(p)
                        
                    elif len(subset) > 1: # Previous body stored and mult ppl detected. Finds detected person who is closest in distance to prev_body. Sets this preson as new body.
                        dist = []
                        for person in subset:
                            coord = get_coords(person, candidate)
                            dist.append(get_average_dist(prev_body, coord))
                        ind = np.argmin(dist) # finds index of minimum distance
                        subset = [subset[ind]] # person who is closest to prev_body
                        subset = np.array(subset)
                        p = get_coords(subset, candidate)
                        prev_body = copy.deepcopy(p)
                    
                    else: # Previous body stored and only one person detected. Sets person as new body.
                        p = get_coords(subset, candidate)
                        prev_body = copy.deepcopy(p)
                        
                    x = get_features(subset, candidate)
                    b_features.append(x) # appends flattened array to body features
                                    
                    ''' Hands '''
                    
                    hands_list = util.handDetect(candidate, subset, frame) # [[99, 305, 137, True], [103, 295, 125, False]]
                    final_peaks = []
                        
                    if len(hands_list) == 0: # Hands not detected by OP detector
                        coord_list = detect_hands(frame)
                        coord_list = sorted(coord_list, key = lambda x: x[4]) # sort by confidence
                        while len(hands_list) < 2 and len(coord_list) > 0:
                            hands_list.append(coord_list[-1])
                            coord_list.pop()
                        
                    else:
                        for hand in hands_list:
                            hand[3] = hand[2] # sets height index (3) equal to width
                
                    all_hand_peaks = []
                    counts = [] # lists number of joints detected for each hand
                    for hand in hands_list:
                        x, y, w, h = bound_to_frame(hand[0], hand[1], hand[2], hand[3], frame)
                        peaks, c = hand_estimation(frame[y:y+h, x:x+w, :])
                        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                        all_hand_peaks.append(peaks)
                        counts.append(c)
                    
                    m1, m2 = 0, 1 # tentative indeces of selected hands in all_hand_peaks
                        
                    if len(all_hand_peaks) != 2:
                        if len(all_hand_peaks) > 2:
                            m1 = np.argmax(counts) # m1, m2 stores indeces of hands with top two most joints
                            counts[h1] = 0
                            m2 = np.argmax(counts)
                        else: # less than two hands
                            while len(all_hand_peaks) != 2:
                                s = [[0,0,n] for n in range(21)] # creates empty hand where n is joint id
                                all_hand_peaks.append(np.array(s))
                                
                    h1 = all_hand_peaks[m1]
                    h2 = all_hand_peaks[m2]
                    final_peaks.append(h1)
                    final_peaks.append(h2)
#                        for k in range(21):
#                            final_peaks.append(h1[k])
#                            final_peaks.append(h2[k])
                                            
                    final_peaks = util.reorder_hands_L_R(final_peaks)
                    h_features.append(np.array(final_peaks).flatten())
                    
                    # Draw body pose
                    canvas = copy.deepcopy(frame)
                    canvas = util.draw_bodypose(canvas, candidate, subset)
                    canvas = util.draw_handpose(canvas, final_peaks)
                    #canvas = util.draw_handpose(canvas, peaks)
                    
                    # Draw and save frame
                    im = plt.imshow(canvas[:, :, [2, 1, 0]])
                    plt.axis('off')
                    frames.append(get_img_from_fig(plt))
                    plt.close()
                
                    #print(str(len(b_features))+'/10')
                
                if len(b_features) == 75: # Breaks after analyzing 75 frames
                    break
            
            body_list[vid_info[1]] = np.array(b_features)
            hand_list[vid_info[1]] = np.array(h_features)
            cap.release()

            # Write video to file
            h, w, layers = frames[0].shape
            out = cv2.VideoWriter(dir+'op/'+str(vid_info[1])+'_op.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w,h))
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()
            
            # Tracks and prints progress through folder
            #print(str(counter),'/',str(len(vid_list)),': ',vid_info[1],' complete',)
            print('{} / {}: {} complete'.format(counter, len(vid_list), vid_info[1]))

        except Exception as e:
            cap.release()
            tb = sys.exc_info()[-1]
            f = tb.tb_frame.f_code.co_filename; l = tb.tb_lineno
            print('{}: An exception occurred in {} at {}. Error: {}'.format(vid_info[1], f, l, repr(e)))
        
    return body_list, hand_list  # Dictionaries with each key storing 2d numpy array with shape (75,72), (75,126)

if __name__ == "__main__":
    dir = 'exp/vids_set/'
    b_data, h_data = get_vid_data(dir) # Dictionaries with keys=filenames, each storing 2d numpy array with shapes (75,72), (75,126)
    scipy.io.savemat('exp/vids_set/set1_body.mat', b_data)
    scipy.io.savemat('exp/vids_set/set1_hand.mat', h_data)
    set_trace()