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
from itertools import product

from detect_hands import detect_hands

import os, sys, inspect, linecache
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

#Body object, estimation
from src import model
from src import util
from src.body import Body

#Hand object, estimation
from src.hand_vid import Hand

#Face
from face.facial_landmarks import detect_facial_keypoints

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

# Returns flattened body feature list to be added to dataset
def get_features(subset, candidate):
    x = []
    subset = subset.flatten()
    for i in range(18): # cycle through each body part
        if int(subset[i]) == -1:
            x.extend([-1,-1,-1,i])
        else:
            x.extend(np.append(candidate[int(subset[i])][0:3],i))
    return x

# Returns coordinate data only, skips score and id
def get_coords(subset, candidate):
    x = []
    subset = subset.flatten()
    for i in range(18):
        if int(subset[i]) == -1:
            x.append([])
        else:
            x.append(candidate[int(subset[i])][0:2])
    return x

# Finds average distance between keypoints for two body subsets
def get_average_dist(prev, curr):
    num, tot = 0, 0
    for i in range(18):
        if all(p == 0 for p in prev[i]) or all(c == 0 for c in curr[i]):
            continue
        px, py, cx, cy = prev[i][0], prev[i][1], curr[i][0], curr[i][1]
        tot += get_distance([px,py], [cx,cy])
#        x, y = px - cx, py - cy
#        tot += math.sqrt(x**2 + y**2)
        num += 1
    if num == 0:
        return 1000
    else:
        return tot/num

# Calculates center of hand box
def get_box_center(box):
    x, y = box[0]+box[2]/2, box[1] + box[2]/2
    return x, y

# Calculates distance between two points
def get_distance(p1, p2):
    x, y = p1[0] - p2[0], p1[1] - p2[1]
    dist = math.sqrt(x**2 + y**2)
    return dist
    
def do_boxes_overlap(box1, box2):
    one_x, one_y, one_w, one_h = box1[0], box1[1], box1[2], box1[3]
    two_x, two_y, two_w, two_h = box2[0], box2[1], box2[2], box2[3]
    if two_x < one_x < two_x+two_w or two_x < one_x+one_w < two_x+two_w:
        if two_y < one_y < two_y+two_h:
            return True
        elif two_y < one_y+one_h < two_y+two_h:
            return True
    return False
    
# set it so that each detected hand finds the tracked hand it's closest to.

def minimize_box_dist(prev, curr, op):
    output = [[0,0,0,0],[0,0,0,0]]; new_tracked = [[], op]
    arr = [[], []] # distances from tracked hand 0 and 1
    ind_no_prev_hand = None

    for i, tracked in enumerate(prev):
        if all(p == 0 for p in tracked):
            ind_no_prev_hand = i
            continue
        for j, detected in enumerate(curr):
            dx, dy = get_box_center(detected)
            tx, ty = get_box_center(tracked)
            dist = get_distance([tx,ty], [dx,dy])
            if op:
                if not do_boxes_overlap(tracked, detected) and dist > tracked[2] and dist > tracked[3]:
                    continue
            arr[i].append([detected, dist, i, j])

    if all(len(p)==0 for p in arr):
        return output, [prev, op]
    
    if ind_no_prev_hand != None:
        candidates = [[t] for t in arr[1-ind_no_prev_hand]]
        for c in candidates:
            c.insert(ind_no_prev_hand, [[0,0,0,0],0,0,0]) # put in place depending on num
    else:
        pairs = list(product(*arr))
        candidates = []
        for pair in pairs:
            if pair[0][3] != pair[1][3]:
                candidates.append(pair)
            
    if not candidates:
        for x in arr:
            for el in x:
                candidates.append(el)
        candidates.sort(key = lambda x: x[1])
        output[candidates[0][2]] = candidates[0][0]
    
    else:
        candidates.sort(key = lambda x: x[0][1]+x[1][1])
        output = [c[0] for c in candidates[0]]
    
    if ind_no_prev_hand != None and not op and len(curr) > 1:
        ind_from_last = -1
        if output[1-ind_no_prev_hand] == curr[-1]:
            ind_from_last = -2
        area_set = output[1-ind_no_prev_hand][2] * output[1-ind_no_prev_hand][3]
        area_unset = curr[ind_from_last][2] * curr[ind_from_last][3]
        if not area_set < (0.25*area_unset) and not area_unset < (0.25*area_set):
            output[ind_no_prev_hand] = curr[ind_from_last]
    
    for i, el in enumerate(output):
        if all(p == 0 for p in el):
            new_tracked[0].append(prev[i])
        else:
            new_tracked[0].append(el)
    
    return output, new_tracked
 
'''
# Returns arrays of hand boxes matched with tracked boxes. If hand is tracked through OP, a returned hand must be less than a width or height's distance of its corresponding tracked hand.
def old_minimize_box_dist(prev, curr, op):
    ret = [[0,0,0,0,1000], [0,0,0,0,1000]]
    ind_from_last = 0
    no_prev_hand = None # holds index value of any [0,0,0,0] placeholder in prev (tracked hand).
    n = 0 # number of previously tracked hands that have been matched
    
    arr = [][]
    
    for i, tracked in enumerate(prev):
        if all(p == 0 for p in tracked):
            continue
        for j, detected in enumerate(curr):
            dx, dy = get_box_center(detected)
            tx, ty = get_box_center(tracked)
            arr[i].append([detected, dist, j])
    
    #pairs = combinations(w_values, 2)
    pairs = [list(zip(each_combination, arr[1])) for each_combination in itertools.combinations(arr[0], len(arr[1]))]
    
    for i, detected in enumerate(curr):
        if n == 2:
            break
        curr_closest = [None, 1000, None]
        for j, hand in enumerate(prev):
            if all(p == 0 for p in hand):
                no_prev_hand = j
                continue
            cx, cy = get_box_center(detected)
            px, py = get_box_center(hand)
            dist = get_distance([px,py], [cx,cy])
            if dist < curr_closest[1]:
                curr_closest = [detected, dist, j]
                ind_from_last = 1 + int(i == len(curr)-1) # 2 if i is last element in curr, 1 if not
        if op:
            if curr_closest[1] > hand[2] or curr_closest[1] > hand[3]: # checks if the distance is less than the width or height of the tracked hand box
                continue
        ret[curr_closest[2]] = curr_closest[0]
        
    if no_prev_hand != None and len(curr) > 1 and not op: # if area of other returned hand is neither 4x larger nor 4x smaller than the remaining highest confidence hand, include this remaining hand
        area_set = ret[no_prev_hand-1][2] * ret[no_prev_hand-1][3]
        area_unset = curr[-1][2] * curr[1][3]
        if not area_set < (0.25*area_unset) and not area_unset < (0.25*area_set):
            ret[no_prev_hand] = curr[-1*(ind_from_last)]
            
    
    for i, hand in enumerate(prev):
        if all(p == 0 for p in hand): # no hand previously stored at index i
            no_prev_hand = i
            continue
        px, py = get_box_center(hand) # center of previous hand
        curr_closest = [None, 1000] # stores closest hand to tracked hand
        for j, detected in enumerate(curr):
            cx, cy = get_box_center(detected) #center of currently analyzed hand
            dist = get_distance([px,py], [cx,cy])
#            x, y = px - cx, py - cy
#            dist = math.sqrt(x**2 + y**2)
            if dist < curr_closest[1]:
                curr_closest = [detected, dist]
                ind_from_last = 1 + int(j == len(curr)-1) # 2 if j is last element in curr, 1 if not
        if op:
            if curr_closest[1] > hand[2] or curr_closest[1] > hand[3]: # checks if the distance is less than the width or height of the tracked hand box
                continue
        if ret[i][4] > curr_closest[1]: # currently stored
            ret[i] = curr_closest[0]
        #remaining.pop(j)
        
    if no_prev_hand and len(remaining) > 0 and not op:
        # if area of other returned hand is neither 4x larger nor 4x smaller than the remaining highest confidence hand, include this remaining hand
        area_set = ret[no_prev_hand-1][2] * ret[no_prev_hand-1][3]
        area_unset = remaining[-1][2] * remaining[1][3]
        if not area_set < (0.25*area_unset) and not area_unset < (0.25*area_set):
            ret[no_prev_hand] = curr[-1*(ind_from_last+1)]
            
    return ret

'''

# Determines whether detected hand is likely a face
def is_hand_face(b, h):
    face = b[14:18] # keypoints values for face
    w_values = []
    for val in face: # keeps only keypoints which are not [0,0]
        if not all(p == 0 for p in val):
            w_values.append(val[:2])
    if not w_values: # no face keypoints detected
        return False
        
    # finds centroid of face keypoints
    x = [p[0] for p in w_values]; y = [p[1] for p in w_values]
    centroid = (sum(x) / len(w_values), sum(y) / len(w_values))
    # if centroid is not within the hand box, return false
    if not h[0] < centroid[0] < h[0]+h[2] or not h[1] < centroid[1] < h[1]+h[3]:
        return False
        
    pairs = combinations(w_values, 2)
    max_dist = 0
    for p in pairs:
        dist = get_distance(p[0], p[1])
        if dist > max_dist:
            max_dist = dist
    if h[2] > 0.7*max_dist or h[3] > 0.7*max_dist: # width or height of hand box is greater than 0.8*max_dist
        return True
    return False

# Returns two dictionaries storing 2d numpy arrays containing pose data for each video in a video set
def get_vid_data(dir, avoid_face, detect_hand_pose=False):

    vid_list = []
    body_list = {}
    hand_list = {}
    body_estimation = Body('../model/body_pose_model.pth')
    hand_estimation = Hand('../model/hand_pose_model.pth')

    if not os.path.exists(dir+'op_new_track'):
        os.makedirs(dir+'op_new_track')

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
        
#        set_trace()
#        vid_info = vid_list[-3]
        
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
            
            start = False
            pos_frame = None
            
            while True:
                
                if start:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,96)
                    start = False
                
                flag, frame = cap.read() # read in the next frame
                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # frame index
                
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
#                        if tracked_hands and tracked_hands[1]: # tracked_hands is tracking an OP detected hand set
#                            tracked_hands = [] # resets tracked_hands
                    
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
                    # take out any selection process (size, distance); look only at the two highest confidence hands and place them
                    
                    
                    frame_hand_data = [[0,0,0,0], [0,0,0,0]]
                    final_hands = []
                    
                    # Retrieve hand box coordinates using OpenPose or Yolo
                    hands_detected = util.handDetect(candidate, subset, frame) # run through OP hand detector
                    
                    
                    # If OP hands are detected, automatically track.
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
                    
                    else:
                        hands_detected = detect_hands(frame) # run through Yolo hand detector
                        hands_detected = sorted(hands_detected, key = lambda x: x[4]) # sort hands by confidence
                        hands_detected = [box[0:4] for box in hands_detected] # remove confidence value at ind 5
                        
                        if avoid_face:
                            for i, hand in enumerate(hands_detected):
                                coooo = get_coords(subset, candidate)
                                if is_hand_face(coooo, hand):
                                    hands_detected[i] = [0,0,0,0]
                        
                        # try with distance tracking non-op
#                        if len(b_features) == 11:
#                            set_trace()
                        
                        if len(tracked_hands) == 0: # no tracked hand
                            frame_hand_data = hands_detected[-2:] # keep two highest confidence hands
                            if len(frame_hand_data) == 2: # if one frame is more then 4x larger than another, keep only the larger
                                a0 = frame_hand_data[0][2] * frame_hand_data[0][3]
                                a1 = frame_hand_data[1][2] * frame_hand_data[1][3]
                                if a0 < (0.25*a1):
                                    frame_hand_data.pop(0)
                                elif a1 < (0.25*a0):
                                    frame_hand_data.pop(1)
                            while len(frame_hand_data) < 2: # If there are less than two hands detected, add up to 2 empty hands to feature array
                                frame_hand_data.append([0,0,0,0])
                            frame_hand_data = util.reorder_hands_L_R(frame_hand_data, detect_hand_pose)
                            #frame_hand_data = minimize_box_dist(tracked_hands[0], hands_detected, tracked_hands[)
                            if not all(p == [0,0,0,0] for p in frame_hand_data):
                                tracked_hands = [frame_hand_data, False]
                            
                        elif len(hands_detected) != 0: # tracked hand and hands were detected
                            frame_hand_data, tracked_hands = minimize_box_dist(tracked_hands[0], hands_detected, tracked_hands[1])
#                            if not all(p == [0,0,0,0] for p in frame_hand_data):
#                                tracked_hands = [frame_hand_data, tracked_hands[1]]
                        '''
                        elif tracked_hands[1] and len(hands_detected) != 0: # tracked hand found through OP and hands are available to test distance
                            frame_hand_data = minimize_box_dist(tracked_hands[0], hands_detected)
                            if not all(p == [0,0,0,0] for p in frame_hand_data):
                                tracked_hands = [frame_hand_data, True]
                        '''

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
                    
                    ''' Face '''
                    face_points = detect_facial_keypoints(frame)
                    
                    ''' Draw '''
                    # Draw body, hand poses/box, face keypoints
                    canvas = copy.deepcopy(frame)
                    canvas = util.draw_bodypose(canvas, candidate, subset)
                    canvas = util.draw_face(canvas, face_points)
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
                
                    print(str(len(b_features))+'/75')
                
                if len(b_features) == 75: # Break after analyzing 75 frames
                    break
            
            # Store body and hand arrays into dictionaries under filename key
            body_list[vid_info[1]] = np.array(b_features).astype(float)
            hand_list[vid_info[1]] = np.array(h_features)
            cap.release()

            # Write video to file
            h, w, layers = frames[0].shape
            if avoid_face:
                out = cv2.VideoWriter(dir+'op_new_track/'+vid_info[1]+'_op.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w,h))
            else:
                out = cv2.VideoWriter(dir+'op_keep_face/'+vid_info[1]+'_op.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w,h))
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()
            
            # Track and print progress through folder
            print('{} / {}: {} complete'.format(counter, len(vid_list), vid_info[1]))
            #set_trace()

        except Exception as e:
            cap.release()
            tb = sys.exc_info()[-1]
            f = tb.tb_frame; l = tb.tb_lineno
            filename = f.f_code.co_filename
            #linecache.checkcache(filename)
            #line = linecache.getline(filename, l, f.f_globals)
            if pos_frame:
                print('{}: An exception occurred in {} at line {} on frame {}. Error: {}'.format(vid_info[1], filename, l, pos_frame, repr(e)))
                #print('Line: "{}"'.format(line))
            else:
                print('Skipped {}'.format(vid_info[1]))
        
    return body_list, hand_list  # Dictionaries with each key storing 2d numpy array with shape (75,72), (75,126) OR (75,8)

if __name__ == "__main__":
    dir = 'exp/single/'
    
    b_data, h_data = get_vid_data(dir, avoid_face = True)
    scipy.io.savemat(dir+dir.split('/')[1]+'_body_new_track.mat', b_data)
    scipy.io.savemat(dir+dir.split('/')[1]+'_hand_new_track.mat', h_data)
    
    set_trace()
