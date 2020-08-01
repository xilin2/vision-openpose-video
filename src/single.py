import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

#Body object
#body_estimation
#-----

from src import model
from src import util
from src.body import Body
#from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
#hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'exp/badPatients/lift_bad_red_L.png'
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)

#im_name = test_image[16:-4]

canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()

features = np.empty(0)
#features = []

if len(subset) != 2:
    print('found ' + str(len(subset)) + ' objects')
    if len(subset) > 2:
        body_parts_count = []
        for person in subset:
            body_parts_count.append(person[19])
        while len(subset) != 2:
            min = body_parts_count.index(min(body_parts_count))
            subset = np.delete(subset, min, axis=0)
            body_parts_count.pop(min)
    else:
        while len(subset) != 2:
            subset = np.append(subset, [np.ones(20)*-1], axis=0)

# assigns feature array per person
x_bound = 0
for person in subset:
    x = []
    x_total = 0
    for i in range(0, 18): # cycle through each body part
        x_total = x_total + candidate[int(person[i])][0]
        if int(person[i]) == -1:
            x.extend([-1,-1,-1,-1])
        else:
            x.extend(candidate[int(person[i])])
    
    # inserts array in proper location depending on 'x' coordinates
    if x_total/person[19] > x_bound:
        features = np.append(features, x)
        #features.append(x)
        x_bound = x_total
    else:
        features = np.insert(features, 0, x)
        #features.insert(0, x)
'''
# accounts for second array in case of undetected person
if len(subset) < 2:
    print('error1')
    features = np.append(features, [np.ones((18,4))*-1], axis=0)
'''

#print(candidate)
#print(subset)
print(features)
print(features.shape)
print(features.ndim)

# detect hand
'''
hands_list = util.handDetect(candidate, subset, oriImg)

all_hand_peaks = []
for x, y, w, is_left in hands_list:
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)
    print(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)
'''
