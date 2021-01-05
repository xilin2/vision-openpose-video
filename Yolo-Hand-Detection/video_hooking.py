import cv2
import numpy as np
import glob
from scipy.io import savemat
from pdb import set_trace
import matplotlib.pyplot as plt
import time

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

from torchvision.models import vgg19
import torchvision.transforms as transforms
#from keras.applications.vgg16 import VGG16
import torch
from fastai.torch_core import flatten_model
from PIL import Image

from src.model import bodypose_model
from src.util import Hook, get_layer_names
from pdb import set_trace

# Returns two dictionaries storing 2d numpy arrays containing pose data for each video in a video set

'''
Modified VGG code with hooks added on non-ReLU layers.
'''
class VGGhook():
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vgg19(pretrained=True).to(self.device)

        self.layers = flatten_model(self.model)
        self.layer_names = get_layer_names(self.layers)
        
        for l in self.layer_names:
            print(l)
        
    def __call__(self, oriImg, layer_to_hook):
        
        # Process image
        img = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        img = Image.fromarray(img)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        data = preprocessing(img)
        
        if self.device.type == 'cuda':
            data = data.cuda() 
        
        try:
            layer_index, layer = self.layer_names[layer_to_hook]
            hook = Hook(self.layers[layer_index])
        
            self.model(data[None, ...])
            hook.close()
            
            #set_trace()
            out = hook.output
            if self.device.type != 'cuda':
                out = out.clone().detach().requires_grad_(True)
            elif self.device.type == 'cuda':
                out = out.cpu().detach().numpy().flatten()

            features = out

            #features = out.detach().numpy().flatten()
        
        except Exception:
            hook.close()
        
        return features

'''
Performs video hooking analysis. Has function to call modified network (either
VGG-19 or Openpose) for each frame in each video. Produces dissimilarity matrix
for each layer (60x60).
'''
class VidHooking():

    def __init__(self, model, dir):
        
        self.model = model
        if model == 'vgg':
            self.net = VGGhook()
            #self.layer_count = len(self.net.layer_names)
            self.layer_count = 17
        elif model == 'openpose':
            self.net = Body('../model/body_pose_model.pth') #calls modified Body code with added hooks
            self.layer_count = 55

        self.vid_list = self.read_vid_data(dir)
            
#        self.body_hooking = Body('../model/body_pose_model.pth')
#        self.body_estimation = Body('../model/body_pose_model.pth', hooking=False)
#        self.frame_list = self.get_frame_info()
    
    def read_vid_data(self, dir):
    
        '''
        Reads in video data for each of 60 videos. Returns a 2d array of format
        [vid info, vid name]
        '''
        
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
    
        '''
        For each video and layer of network, finds average of layer output
        across all frames. Returns dictionary of 60x60 rdms (one for each layer).
        '''
        
        rdms = {} # Eventually store rdms
        
        #layers_of_interest = [14, 17, 24, 31, 38, 45, 52]
        
        # Cycles through each hooking layer
        #for layer in range(0,56):
        for layer in range(self.layer_count):
        
            #print('working on layer {}/{}'.format(layer+1, self.layer_count))
            
            '''
            if layer not in layers_of_interest: # visualize
                continue
            '''
            
            #print(layer) #visualize
 
            avgd_data_for_layer = []
            
            # Cycles through each video in vid_list
            for counter, vid_info in enumerate(self.vid_list, 1): # vid_info[video, filename]

#                if counter is not 4: # visualize
#                    continue
                data = self.get_vid_data_for_layer(vid_info, layer)
#                continue # visualize
                
                avgd_data_for_layer.append(data)
        
            # removed (released) temporarily for visualization
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
            #while frames_read < 1:
            
                flag, frame = cap.read() # read in the next frame
                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # frame index

                if not flag: # The next frame is not ready, so we try to read it again
                   cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                   cv2.waitKey(1000)

                elif pos_frame not in skip_val: # Frame number is not skipped
                    
                    # first frame, sets the shape for the proceeding array additions
                    if frames_read == 0:
                        if layer < 55:
                                
                            return_data = np.array(self.net(frame, layer_to_hook=layer))
                            
                            '''
                            print(sum(return_data.flatten()))
                            
                            # visualize data
                        
                            probMap = return_data[0,:,:,:]
                            
                            for r in probMap:
                                map = cv2.resize(r, (512,512))
                                plt.imshow(map, alpha=0.6)
                            plt.savefig('layer_images/vid4_{}_NO14'.format(layer))
                            
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            
                            return return_data
                            '''
                        
                        '''
                        else:
                            candidate, subset = self.body_estimation(frame)
                            return_data = np.array(self.get_final_pose(candidate, subset, frame.shape[0]))
                        '''
                    else:
                        if layer < 55:
                            features = np.array(self.net(frame, layer_to_hook=layer))
                            return_data = np.add(return_data, features)
                        
                        '''
                        else:
                            candidate, subset = self.body_estimation(frame)
                            features = np.array(self.get_final_pose(candidate, subset, frame.shape[0]))
                        '''
            
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
    #vgghook = VGGhook()
    
    start_time = time.time()
    
    hooking = VidHooking('openpose', dir)
    rdms = hooking.get_full_features()
    savemat(dir+dir.split('/')[1]+'_hooking_vgg19_RDMs_NEW.mat', rdms)
    
    print("--- Hooking completed in %s minutes ---" % ((time.time() - start_time)/60))
    
