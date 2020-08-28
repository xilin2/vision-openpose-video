import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import math

import scipy.cluster.hierarchy as sch
from scipy.stats import zscore, pearsonr

from pdb import set_trace

# Returns a set of 26 trajectories (one for each point describing a pose), each with 75 values. (0,0) points are set to the value of the previous point so to prevent misshapen trajectories due to the numerous (0,0)s scattered throughout.
def get_trajectories(vid_data):
    
    n = 26 # number of trajectories / points describing each video
    
    trajs = [[] for i in range(n)] # empty array to store 26 trajectories, each with 75 values
    traj_incl_0 = [[] for i in range(n)] # empty array to store un-modified trajectories array (eg. still has (0,0) points)
    separated_by_frame = [vid_data[i:i+2*n] for i in range(0, 75*n*2, n*2)] # separates vid_data into 75 frames, each with 26*2 = 52 values
    for i in range(len(separated_by_frame)): # separates each frame into 26 points of (x,y)
        points = [separated_by_frame[i][j:j+2] for j in range(0, n*2, 2)]
        separated_by_frame[i] = points
    for frame in separated_by_frame: # runs through each frame, separating the points into their trajectories
        for i in range(n):
            traj_incl_0[i].append(frame[i])
    for i, tr in enumerate(traj_incl_0): # tr = traj of a single point (len=75)
        for j, pnt in enumerate(tr):
            if all(coord == 0 for coord in pnt): # point = (0,0)
                if j == 0: # if first point is (0,0), then set point to first non-zero point in trajectory
                    non_zeros = [p for p in tr if any(el != 0 for el in p)]
                    if len(non_zeros) == 0:
                        trajs[i].append([0,0])
                    else:
                        trajs[i].append(non_zeros[0])
                else: # if not first point, set point to value of previous point
                    trajs[i].append(trajs[i][j-1])
            else:
                trajs[i].append(pnt)
    
    return trajs

# Takes two matrices A and B. Finds the Procrustes distance using sqrt(sum((A-B)^2))
def find_procrustes_distance(mtx1, mtx2):
    squared_distances = []
    for i in range(len(mtx1)):
        s = (mtx1[i][0]-mtx2[i][0])**2 + (mtx1[i][1]-mtx2[i][1])**2
        squared_distances.append(s)
    return math.sqrt(sum(squared_distances))

# Combines body and hand datasets into one dataset, with hand data appended onto the end of each frame.
def combine_body_hand(body, hand):

    comb_data_1 = {}; comb_data_2 = {}
    for key in body:
        if key.split('_')[0][-1] == '1':
            comb_data_1[key] = body[key]
            comb_data_1[key].extend(hand[key])
        else:
            comb_data_2[key] = body[key]
            comb_data_2[key].extend(hand[key])
                
    return comb_data_1, comb_data_2

# Converts (x,y,w,h) hand box to set of 4 corner coordinates
def convert_hand_to_coord(hand):
    coords = []
    coords.append(hand[0]); coords.append(hand[1]) #URH
    coords.append(hand[0]); coords.append(hand[1]+hand[3]) #LRH
    coords.append(hand[0]+hand[2]); coords.append(hand[1]) #ULH
    coords.append(hand[0]+hand[2]); coords.append(hand[1]+hand[3]) #LLH
    return coords

# Averages hand box coordinates across 75 frames. Ignores hand boxes which are (0,0,0,0) which represent an undetected hand. Each frame contains 8 hand box coordinates.
def average_hands(video):
    
    avgd = [[] for i in range(16)]
    for frame in video:
        for i, hand in enumerate(frame):
            if all(v == 0 for v in hand):
                  continue
            for j, val in enumerate(hand):
                  avgd[j+8*i].append(val)
    for i in range(len(avgd)):
        if len(avgd[i]) == 0:
            avgd[i] = 0
        else:
            avgd[i] = np.mean(avgd[i])

    return avgd
    
# Averages body coordinates across 75 frames. Ignores body points which are (0,0) which represent an undetected point. Each frame contains 18 body coordinates.
def average_body(video):
    
    avgd = [[] for i in range(len(video[0]))]
    for frame in video:
        for i in range(0, len(frame), 2):
            if int(frame[i]) == 0 and int(frame[i+1]) == 0:
                continue
            avgd[i].append(frame[i]); avgd[i+1].append(frame[i+1])
    for i in range(len(avgd)):
        if len(avgd[i]) == 0:
            avgd[i] = 0
        else:
            avgd[i] = np.mean(avgd[i])

    return avgd
    
# Returns hands video data set, organized by frame and with hand box data (x,y,w,h) converted to four coordinates (len=75).
def clean_hands(video):
    clnd = [[] for i in range(len(video))]
    for j, frame in enumerate(video):
        separated_by_hand = [frame[i:i+4] for i in range(0, 8, 4)]
        for i, hand in enumerate(separated_by_hand):
            clnd[j].append(convert_hand_to_coord(hand))
    return clnd

# Returns body video data set, organized by frame and with score and ID removed (len=75).
def clean_body(video):
    clnd = [[] for i in range(len(video))]
    for j, frame in enumerate(video):
        for i, val in enumerate(frame):
            if (i%4) == 3 or (i%4) == 2:
                continue
            clnd[j].append(val)
    return clnd

# Returns a vector of all pairs that do not include the video index in a dissimilarity matrix, and a vector of all pairs that do include the video index.
def hold_pairs(hold, matrix):
    held_values = []
    vector = []
    # Moves row by row through top triangle of distance matrix. Rows get shorter and shorter as loop travels down the triangle.
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            if i is hold or j is hold:
                held_values.append(matrix[i][j])
                continue
            vector.append(matrix[i][j])
    return vector, held_values

# Return unraveled dataframes with body + hand features. The index corresponds to a value within each video's dataset, and the columns correspond to the videos in the video set.
def to_df(data, names):
    
    actions = []
    values = [[] for i in range(len(data[list(data.keys())[0]]))]
    
    for key in data:
        i = names[names['Video name']==key+'.mp4']
        if i.empty:
            continue
        actions.append(i.iloc[0]['Action Name'])
        for i, val in enumerate(data[key]):
            values[i].append(val)
    
    df = pd.DataFrame(values, columns=actions)
    df = df.apply(zscore)
    
    return df, actions
    
# Plot and save dendrogram produced from hierarchical clustering analysis (clustering method provided in argument). Returns hierarchical clustering solution as a list.
def plot_heir_cluster(df, met, dir, vid_set_num):
    link = sch.linkage(df, method=met)
    dg = sch.dendrogram(link, orientation="right", leaf_font_size=6, labels=df.index)
    plt.title('Video Set {}: No Tracking'.format(vid_set_num))
    plt.xlabel('Videos')
    plt.gcf().set_size_inches((10,10))
    plt.subplots_adjust(left=0.22)
    plt.savefig('{}/{}_vidset{}_dgram.png'.format(dir, met, vid_set_num))
    #plt.show()
    plt.close()
    return dg['leaves']

# Plot and save correlation heatmap of dataframe
def plot_heatmap(df, met, dir, vid_set_num):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap)
    plt.title('Video Set {}'.format(vid_set_num), fontsize=20)
    plt.subplots_adjust(left=0.22, bottom=0.20, top=0.95, right=0.90)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('{}/{}_vidset{}_heatmap.png'.format(dir, met, vid_set_num))
    #plt.show()
    plt.close()
    return

# Randomly shuffles dataset and correlates first half of shuffled set to second half of shuffled set. Repeats 'num' times. Used to estimate reliability of dataset.
def get_split_half_reliabilities(num, dataset):
    
    reliabilities = {}
    for vid in dataset:
        corr_sum = 0
        for i in range(num):
            shuffled = random.sample(dataset[vid], len(dataset[vid]))
            half1 = shuffled[:37]; half2 = shuffled[37:74]
            half1 = np.array(half1, dtype='float').flatten()
            half2 = np.array(half2, dtype='float').flatten()
            corr_sum += pearsonr(half1, half2)[0]
        reliabilities[vid] = corr_sum/num
    return reliabilities

# Plots error bar given the data set of prediction accuracy scores from cross-validation.
def plot_error_bar(scores):
    
    set_trace()
    
    fig, ax = plt.subplots()
    labels = ['']
    for model in scores:
        for behavior in scores[model]:
            labels.append('{}-{}'.format(model, behavior))
    ax.set_ylabel('Correlation')
    ax.set_title('Comparison of Average and Procrustes distance cross-validation accuracies')
    ax.axhline(y=0, color='black')
    ax.set_xticks(np.arange(len(labels)))
    ax.axhline(y=0, color='black')
    ax.set_xticklabels(labels)
    x_pos = 1
    
    colors = ['black', 'blue']
    
    for i, model in enumerate(scores):
        for behavior in scores[model]:
        
            s = scores[model][behavior]
        
            mean = np.mean(s)
            bar_middle = 0.5*(np.percentile(s,25) + np.percentile(s,75))
            bar_height = np.percentile(s,75) - np.percentile(s,25)
            median = np.percentile(s,50)
        
            ax.barh(y=bar_middle, height=bar_height, width=0.5, left=x_pos-0.25, color='white', edgecolor=colors[i])
            ax.plot([x_pos-0.245, x_pos+0.245], [median, median], marker=None, c=colors[i], linewidth=1)
            ax.plot([x_pos, x_pos], [min(s), max(s)], marker=None, c=colors[i], linestyle='dashed', linewidth=1, zorder=0)
            ax.plot([x_pos-.12, x_pos+.12], [min(s), min(s)], marker=None, c=colors[i], linewidth=1)
            ax.plot([x_pos-.12, x_pos+.12], [max(s), max(s)], marker=None, c=colors[i], linewidth=1)
            ax.plot([x_pos-.245, x_pos+.245], [mean, mean], c='red', marker=None, linewidth=1)
            x_pos += 1
    
    plt.show()
