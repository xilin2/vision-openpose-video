import cv2
import json
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage.measure import label
from pdb import set_trace

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.model import handpose_model
from src import util
#from model import handpose_model
#import util

import copy

import itertools
import statistics

class Hand(object):
    def __init__(self, model_path, sep=False):
        self.sep = sep
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        
        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                output = self.model(data).cpu().numpy()
                # output = self.model(data).numpy()q

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)

        palm_ind = [0, 5, 9, 13, 17, 2]
        #all_peaks = [[] for i in range(len(palm_ind))]
        all_peaks = []
        
        if self.sep:
            count = 0
            for part in range(21):
                map_ori = heatmap_avg[:, :, part]
                one_heatmap = gaussian_filter(map_ori, sigma=3)
                binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
                # 全部小于阈值
                if np.sum(binary) == 0:
                    all_peaks.append([0, 0, part])
                    continue
                label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)

                max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1 # find second largest by setting array[max_index] to 0
                label_img[label_img != max_index] = 0
                map_ori[label_img == 0] = 0

                y, x = util.npmax(map_ori)
                all_peaks.append([x, y, part])
                count += 1
            
            return np.array(all_peaks), count
        
        #for i, part in enumerate(palm_ind):
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
           # 全部小于阈值
            if np.sum(binary) == 0:
                all_peaks.append([0, 0, part])
                all_peaks.append([0, 0, part])
                continue
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            scores = [np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]
            max_index = np.argmax(scores) + 1
            if label_numbers > 1:
                scores[max_index-1] = 0
                max_2_index = np.argmax(scores) + 1
                label_2_img = copy.deepcopy(label_img)
                map_2_ori = copy.deepcopy(map_ori)
                label_2_img[label_2_img != max_2_index] = 0
                map_2_ori[label_2_img == 0] = 0
                y, x = util.npmax(map_2_ori)
                all_peaks.append([x, y, part])
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0
            y, x = util.npmax(map_ori)
            all_peaks.append([x,y,part])
            if label_numbers < 2:
                all_peaks.append([0,0, part])
        
        return all_peaks
        
        '''
        combos = list(itertools.product(*all_peaks[1:-1])) # creates combos of knuckle point data to test
        xys = []
        for c in combos: # tranposes each combos array so to separate x and y points
            xys.append([list(x) for x in zip(*c)])
        
        # find r of each combo
        correlations_xy = [np.corrcoef(points[0], points[1])[0,1] for points in xys]
        r = [abs(v) for v in correlations_xy]
        set = []
        counter = 0
        set_trace()
        while True: # finds sets of knuckle points with highest r
            max_r = np.argmax(r)
            if r[max_r] > .1 and counter < 2:
            #if counter < 2:
                if isnotin(list(combos[max_r]), set):
                    set.append(list(combos[max_r]))
                    counter += 1
                r[max_r] = 0
            else:
                break
        # find wrist point that matches
    
        return set
    
        #return all_peaks

      # set_trace()
        all_peaks = []
        peak_counter = 0 #added
        
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
           
      #  set_trace()
        return([all_peaks[0], all_peaks[5], all_peaks[9], all_peaks[13], all_peaks[17], all_peaks[2]])
        
        
        # find connection in the specified sequence,
        edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
        [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        
        #palm_edge = [[0, 5], [0, 9], [0, 13], [0, 17], [0, 2]]
        palm_ind = [0, 5, 9, 13, 17, 2]
        palm_edge = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]  # wrt palm_ind
        
        subset = []
        
        # find all possibilties for palm connections, and choose one from each connection that gives that set the smallest standard deviation with length
        
        hand_p = {}
            
        palm = []
        for ind, point in enumerate(palm_ind):
            palm.append(all_peaks[point]) # creates
        palm_combo = list(itertools.product(*palm)) # creates combo of diff point data
        palm_dists = self.find_distances(points, palm_edge)) for points in palm_combo
        palm_pst = statistics.pstdev(dist) for dist in palm_dists
        max_i = index(max(palm_pst)
        curr_palm = palm_combo[max_i]
        if palm_pst(max_i) > .25 * statistics.mean(palm_dists[max_i]) or :
            break
        points = list(points)
            dist = self.find_distances(points, palm_edge))
            st.append([statistics.pstdev(dist), statistics.mean(dist)])
        curr_palm = palm_combo(index(max(stdv)))
            
            
            
            for i, ind in enumerate(palm_ind):
                palm[ind] = curr_palm[i]
            if mean(
            # if standard deviation is less than 1/4 the average. and avg score is above .1
            # for later ones, if mean is less than palm mean
        
            candB = all_peaks[palm[k][1]]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = palm[k]
        
        '''
            
       # return np.array(all_peaks)
       
def isnotin(a, b):
    if len(b) != 0:
        b = b[0]
    for i in a:
        for k in b:
            if i == k:
                return False
    return True

def find_distances(points, edges):
    dist = []
    for i, pair in enumerate(edges):
        pA = points[pair[0]]
        pB = points[pair[1]]
        xd = pB[0] - pA[0]
        yd = pB[1] - pA[1]
        dist[i] = sqrt(xd**2 + yd**2)

def draw_points(canvas, points):
    fig = Figure(figsize=plt.figaspect(canvas))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)
    
    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
        
    colors = ['r.', 'b.', 'g.', 'y.', 'k.', 'm.']

    for point in points:
        x = point[0]
        y = point[1]
        ax.plot(x, y, 'r.')
        #ax.text(x, y, str(i)+': '+str(k))
    
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas
        
if __name__ == "__main__":
    hand_estimation = Hand('../model/hand_pose_model.pth')

    # test_image = '../images/hand.jpg'
    test_image = '../images/test2.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    peaks = hand_estimation(oriImg)
    #canvas = util.draw_handpose(oriImg, peaks, True)
    set_trace()
    canvas = draw_points(oriImg, peaks)
    cv2.imshow('', canvas)
    cv2.waitKey(0)

