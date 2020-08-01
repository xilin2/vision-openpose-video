# import body object, estimation packages
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob # load image folder
import csv

# ----- Body object, body_estimation
from src import model
from src import util
from src.body import Body
#from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
#hand_estimation = Hand('model/hand_pose_model.pth')

image_list = []
error_list = [] # track which images do not detect exactly 2 figures

# parameters to be sent to SVM
features_list = np.empty((0,144))
labels_list = [] # right or left
groups_list = [] # action taken

# cycle through folder to retrieve/store name and image data
for filename in glob.glob('exp/proPatients/*.png'):
    im = cv2.imread(filename)
    image_list.append([im, filename[16:-4]])
    #image_list.append([im, filename[9:-4]])
    image_list.sort(key = lambda x: x[1]) # alphabetical order

# gather features, label, and group info for each image
for im_info in image_list: # im_info[image, filename]
    
    name = im_info[1]
    print(name) # track progress through folder
    
    candidate, subset = body_estimation(im_info[0])
        
    canvas = copy.deepcopy(im_info[0])
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    # produce and save plot
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.savefig('exp/proPatients/plots/'+im_info[1]+'_plot.png')
    #plt.show()
    plt.close()
        
    features = np.empty(0)
    
    # edit subset if wrong num of ppl detected
    if len(subset) != 2:
        error_list.append([len(subset), im_info[1]])
        print('error' + str(len(subset)) + ': ' + im_info[1])
    
        if len(subset) > 2:
            body_parts_count = []
            for person in subset:
                body_parts_count.append(person[19])
            while len(subset) != 2:
                smallest = body_parts_count.index(min(body_parts_count))
                subset = np.delete(subset, smallest, axis=0)
                body_parts_count.pop(smallest)
        else:
            while len(subset) != 2:
                subset = np.append(subset, [np.ones(20)*-1], axis=0)
            
    # produces feature array for each detected person
    x_bound = 0 # variable to aid in determining left/right
    for person in subset:
        coord = [] # array to store coordinate data
        x_total = 0 # x_coord sums
        for i in range(0, 18): # cycle through each body part
            x_total = x_total + candidate[int(person[i])][0] # adds up x-coords
            
            if int(person[i]) == -1:
                coord.extend([-1,-1,-1,-1])
            else:
                coord.extend(candidate[int(person[i])])

        # inserts array in proper location depending on 'x' coordinates
        if x_total/person[19] > x_bound: # finds average of x-coord
                                            # will always be true for subset[0]
            features = np.append(features, coord, axis=0)
            x_bound = x_total
        else:
            features = np.insert(features, 0, coord, axis=0)
            
    '''
    # accounts for second array in case of undetected person
    if len(subset) < 2:
        error_list.append([len(subset), im_info[1]])
        print('error1: ' + im_info[1])
        features = np.append(features, [np.ones((18,4))*-1], axis=0)
    '''
        
    features_list = np.append(features_list, [features], axis=0)
    
    labels_list.append(name[-1])
    groups_list.append(name[:name.find('_')])

#print(features_list)
#print(labels_list)
#print(groups_list)
print(features_list.ndim)
print(features.shape)

with open('exp/proPatients/error.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerows(error_list)
    
# --- end of Body object, body_estimation

# Learning

from svm import group_kfold

group_kfold(features_list, np.array(labels_list), np.array(groups_list), 27, 'proPatients')
    #np.array(groups_list), 3, 'demo')

'''
with open('exp/badPatients/results.csv', 'w', newline='') as file:
   
    writer = csv.writer(file, delimiter=';')
    
    image_list = []
    error_list = [] # track which images without exactly 2 figures
    
    # parameters to be sent to SVM
    features_list = np.empty(0,0,18,4) #
    labels_list = np.empty(0)
    groups_list = np.empty(0)
    
    
    for filename in glob.glob('exp/badPatients/*.png'):
        im = cv2.imread(filename)
        image_list.append([im, filename[16:-4]])
        print(filename[16:-4])
        
    for im_info in image_list: # where im_info[image, name]
        candidate, subset = body_estimation(im_info[0])
        
        canvas = copy.deepcopy(im_info[0])
        canvas = util.draw_bodypose(canvas, candidate, subset)
        
        miss = len(subset)*18-len(candidate) # expected parts - found parts
      
        if miss < 0: # more parts than expected for detected body count
            if len(subset) < 2:
                err = 1 # detects less than two people but too many parts
            elif len(subset) == 2:
                err = 2 # detects two people but more than 36 parts
            else:
                err = 3 # other
            error_list.append([err, im_info[1]])
            #print('error' + str(err) + ': ' + im_info[1])
            continue
        
        elif len(subset) != 2:
            err = 4 # all okay other than too many/little detected bodies
            error_list.append([err, im_info[1]])
            #print('error' + str(err) + ': ' + im_info[1])
            continue
       
        else:
            to_add = np.zeros((miss, 4))
            writer.writerows([im_info[1], subset, np.append(candidate, to_add, axis=0)])

        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.savefig('exp/badPatients/plots/'+im_info[1]+'_plot.png')
        #plt.show()
        plt.close()

'''
