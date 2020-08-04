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
from itertools import combinations

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
def bound_to_frame(box_coord, frame):

    x, y, w, h = box_coord[0], box_coord[1], box_coord[2], box_coord[3]
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
        
    return [x, y, w, h]
        

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
            x.extend([-1,-1,-1,i])
        else:
            x.extend(np.append(candidate[int(subset[i])][0:3],i))
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
        
def get_box_center(box):
    x, y = box[0]+box[2]/2, box[1] + box[2]/2
    return x, y

def get_distance(p1, p2):
    x, y = p1[0] - p2[0], p1[1] - p2[1]
    dist = math.sqrt(x**2 + y**2)
    return dist

def minimize_box_dist(prev, curr):
    ret = [[0,0,0,0], [0,0,0,0]]
    remaining = [i for i in curr] # tracks remaining unassigned hands
    no_prev_hand = None
    for i, hand in enumerate(prev):
        if all(p == 0 for p in hand): # no hand stored
            no_prev_hand = i
            continue
        #px, py = hand[0]+hand[2]/2, hand[1] + hand[2]/2 # coordinates of box center
        px, py = get_box_center(hand)
        curr_closest = [None, 1000] # stores closest hand to tracked hand
        for j, detected in enumerate(curr):
            #cx, cy = detected[0]+detected[2]/2, detected[1] + detected[2]/2
            cx, cy = get_box_center(detected)
            x, y = px - cx, py - cy
            dist = math.sqrt(x**2 + y**2)
            if dist < curr_closest[1]:
                curr_closest = [detected, dist]
        if curr_closest[1] < hand[2] or curr_closest[1] < hand[3]: # checks if the distance is less than the width or height of the tracked hand box
            ret[i] = curr_closest[0]
            remaining.pop(j)
#    if no_prev_hand and len(remaining) > 0:
#        ret[no_prev_hand] = remaining[-1] # sets missing hand as the remaining hand with the highest confidence
    
    return ret
            
def is_hand_face(b, h):
    face = b[14:18]
    w_values = []
    for val in face:
        if not all(p == 0 for p in val):
            w_values.append(val[:2])
    if not w_values: # no face
        return False
    x = [p[0] for p in w_values]; y = [p[1] for p in w_values]
    centroid = (sum(x) / len(w_values), sum(y) / len(w_values))
    if centroid[0] < h[0] or centroid[0] > h[0]+h[2]: # center of face does not lie in hand box
        if centroid[0] < h[1] or centroid[0] > h[1]+h[3]:
            return False
    pairs = combinations(w_values, 2)
    max_dist = 0
    for p in pairs:
        dist = get_distance(p[0], p[1])
        if dist > max_dist:
            max_dist = dist
    if h[2] > 0.7*max_dist or h[3] > 0.7*max_dist:
        return True
    return False

# Returns two dictionaries storing 2d numpy arrays containing pose data for each video in a video set
def get_vid_data(dir, avoid_face, detect_hand_pose=False):

    vid_list = []
    body_list = {}
    hand_list = {}
    body_estimation = Body('../model/body_pose_model.pth')
    hand_estimation = Hand('../model/hand_pose_model.pth')

    if not os.path.exists(dir+'op_avoid_face'):
        os.makedirs(dir+'op_avoid_face')
    if not os.path.exists(dir+'op_keep_face'):
        os.makedirs(dir+'op_keep_face')

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
            # 75 = lowest number of total frames in set1; 78 = lowest number of total frames in set2
            if not tot_frame == 75:
                skip_mult = tot_frame/(tot_frame - 75) # Because videos have different frame lengths, this variable is used to skip over multiples of its value so that after having read through the video in its entirety, only 75 frames have been analyzed.
                skip_val = [round(skip_mult*(i+1)) for i in range(tot_frame-75)] # Stores frame indexes to be skipped
            else:
                skip_val = []
            
            length = tot_frame / cap.get(cv2.CAP_PROP_FPS)
            fps = 75/length

            b_features = [] # 2d array to temporarily store each video's pose features, dim = [frame count=75), 18*4=72]
            h_features = [] # dim = [75, 3*21*2=126] OR [75, 4*2=8]
            frames = [] # Stores drawn frames to later write into video

            tracked_body = [] # Stores the last detected body pose
            tracked_hands = [] # Stores tracked hands
            
            pos_frame = None
            
            while True:
                
                flag, frame = cap.read() # read in the next frame
                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # frame index
                
                # cap.set(cv2.CAP_PROP_POS_FRAMES,55)
                
                if not flag: # The next frame is not ready, so we try to read it again
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                    cv2.waitKey(1000)
                
                elif pos_frame not in skip_val: # Frame number is not skipped
                    
                    ''' Body '''
                    
                    candidate, subset = body_estimation(frame)
                    
                    if len(subset) < 1: # No person detected. Create an empty features array and reset tracked body to []
                        while len(subset) != 1:
                            subset = np.append(subset, [np.ones(20)*-1], axis=0)
                        tracked_body = [] # resets tracked_body
                        if tracked_hands and tracked_hands[1]: # tracked_hands is tracking an OP detected hand set
                            tracked_hands = [] # resets tracked_hands
                    
                    elif len(tracked_body) == 0: # No tracked body stored, more than 1 person detected. Set person with most keypoints as new tracked body.
                        if len(subset) > 1: # Reduce subset to person with most keypoints
                            subset = sorted(subset, key = lambda x: x[19])
                            subset = np.array([subset[len(subset)-1]])
                        p = get_coords(subset, candidate)
                        tracked_body = copy.deepcopy(p)
                    
                    elif len(subset) > 1: # There is a body being tracked and mult ppl are detected. Find detected person who is closest in distance to tracked_body. Set this person as new tracked body.
                        dist = []
                        for person in subset:
                            coord = get_coords(person, candidate)
                            dist.append(get_average_dist(tracked_body, coord))
                        ind = np.argmin(dist) # find index of minimum distance
                        subset = np.array([subset[ind]]) # person who is closest to tracked_body
                        p = get_coords(subset, candidate)
                        tracked_body = copy.deepcopy(p)
                    
                    else: # Tracked body stored and only one person detected. Set person as new tracked body.
                        p = get_coords(subset, candidate)
                        tracked_body = copy.deepcopy(p)
                        
                    final_body = get_features(subset, candidate)
                    b_features.append(final_body) # append flattened array to body features
                                    
                    ''' Hands '''
                    
                    frame_hand_data = [[0,0,0,0], [0,0,0,0]]
                    final_hands = []
                    
                    # Retrieve hand box coordinates using OpenPose or Yolo
                    hands_detected = util.handDetect(candidate, subset, frame) # run through OP hand detector
                    
                    # If OP hands are detected, automatically track. Otherwise, send through Yolo. If no previously tracked hands, take the hands with the highest confidence. If hands are tracked by
                    
                    if len(hands_detected) != 0:
                        for hand in hands_detected:
                            hand[0] = hand[0] + hand[2]/4 # shrink box size by 50%
                            hand[1] = hand[1] + hand[2]/4
                            hand[2] = hand[2]/2
                            if hand[3]: # if left-handed, put in 0 index of frame_hand_data
                                hand[3] = hand[2] # set height index (3) equal to width
                                frame_hand_data[0] = hand
                            else: # put in 1 index of frame_hand_data
                                hand[3] = hand[2]
                                frame_hand_data[1] = hand
                        tracked_hands = [frame_hand_data, True]
                            
                    else: # no hands detected by OP hand detector
                    
                        hands_detected = detect_hands(frame) # run through Yolo hand detector
                        hands_detected = sorted(hands_detected, key = lambda x: x[4]) # sort hands by confidence
                        hands_detected = [box[0:4] for box in hands_detected] # remove confidence value at ind 5
                        if avoid_face:
                            for i, hand in enumerate(hands_detected):
                                coooo = get_coords(subset, candidate)
                                if is_hand_face(coooo, hand):
                                    hands_detected[i] = [0,0,0,0]
                        if len(tracked_hands) == 0 or not tracked_hands[1]: # no previous hand detected, or previous hand detected through Yolo
                            frame_hand_data = hands_detected[-2:] # keep two highest confidence hands
                            while len(frame_hand_data) < 2: # If there are less than two hands detected, add empty hands to feature array
                                frame_hand_data.append([0,0,0,0])
                            frame_hand_data = util.reorder_hands_L_R(frame_hand_data, detect_hand_pose)
                            tracked_hands = [frame_hand_data, False]
                        elif tracked_hands[1] and len(hands_detected) != 0: # tracked hand found through OP and hands are available to
                            frame_hand_data = minimize_box_dist(tracked_hands[0], hands_detected)
                            if not all(p == [0,0,0,0] for p in frame_hand_data):
                                tracked_hands = [frame_hand_data, True]

                    # detect_hand_pose is True. Send boxed portions of image into OP hand pose network to detect hand pose
                    if detect_hand_pose:
                        counts = [] # list number of keypoints detected for each hand
                        for hand in frame_hand_data:
                            if all(p == 0 for p in hand):
                                final_hands.append(np.array([[0,0,n] for n in range(21)]))
                            else:
                                x = hand[0]; y = hand[1], w = hand[2], h = hand[3]
                                peaks, c = hand_estimation(frame[y:y+h, x:x+w, :])
                                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                                final_hands.append(peaks)
                    
                    else:
                        final_hands = frame_hand_data
   
                    h_features.append(np.array(final_hands).flatten())
                    
                    # Draw body and hand poses/box
                    canvas = copy.deepcopy(frame)
                    canvas = util.draw_bodypose(canvas, candidate, subset)
                    if detect_hand_pose:
                        canvas = util.draw_handpose(canvas, final_hands)
                    else:
                        final_hands = [map(int,i) for i in final_hands]
                        canvas = util.draw_handbox(canvas, final_hands)
                    
                    # Draw and save frame
                    im = plt.imshow(canvas[:, :, [2, 1, 0]])
                    plt.axis('off')
                    frames.append(get_img_from_fig(plt))
                    plt.close()
                
                    #print(str(len(b_features))+'/75')
                
                if len(b_features) == 75: # Break after analyzing 75 frames
                    break
            
            # Store body and hand arrays into dictionaries under filename key
            body_list[vid_info[1]] = np.array(b_features).astype(float)
            hand_list[vid_info[1]] = np.array(h_features)
            cap.release()

            # Write video to file
            h, w, layers = frames[0].shape
            if avoid_face:
                out = cv2.VideoWriter(dir+'op_avoid_face/'+vid_info[1]+'_op.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w,h))
            else:
                out = cv2.VideoWriter(dir+'op_keep_face/'+vid_info[1]+'_op.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w,h))
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()
            
            # Track and print progress through folder
            print('{} / {}: {} complete'.format(counter, len(vid_list), vid_info[1]))

        except Exception as e:
            cap.release()
            tb = sys.exc_info()[-1]
            f = tb.tb_frame.f_code.co_filename; l = tb.tb_lineno
            if pos_frame:
                print('{}: An exception occurred in {} at line {} on frame {}. Error: {}'.format(vid_info[1], f, l, pos_frame, repr(e)))
            else:
                print('Skipped {}'.format(vid_info[1]))
        
    return body_list, hand_list  # Dictionaries with each key storing 2d numpy array with shape (75,72), (75,126) OR (75,8)

if __name__ == "__main__":
    dir = 'exp/demo/'
    
    b_f_data, h_f_data = get_vid_data(dir, avoid_face = True)
    scipy.io.savemat(dir+dir.split('/')[1]+'_body_avoid_face.mat', b_f_data)
    scipy.io.savemat(dir+dir.split('/')[1]+'_hand_avoid_face.mat', h_f_data)
    
    set_trace()
    
    b_data, h_data = get_vid_data(dir, avoid_face = False) # Dictionaries with keys=filenames, each storing 2d numpy array with shapes (75,72), (75,126)
    scipy.io.savemat(dir+dir.split('/')[1]+'_body.mat', b_data)
    scipy.io.savemat(dir+dir.split('/')[1]+'_hand.mat', h_data)
    
    set_trace()
