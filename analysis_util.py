import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np
import pandas as pd
import seaborn as sns
import random
import math

import scipy.cluster.hierarchy as sch
from scipy.stats import zscore, pearsonr, wilcoxon

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
                #print('yes')
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

# Combines body and hand datasets into one dataset, with hand data appended onto the end of each frame (or in the case if 'avg', at the end of each video). Final array is flattened
def combine_body_hand(body, hand):

    comb_data_1 = {}; comb_data_2 = {}
    
    for key in body:
        if key.split('_')[0][-1] == '1':
            comb_data_1[key] = []
            # if data is split into frames
            if isinstance(body[key][0], (list, np.ndarray)):
                for i in range(len(body[key])):
                    comb_data_1[key].extend(body[key][i])
                    comb_data_1[key].extend(np.ravel(hand[key][i]))
            else:
                comb_data_1[key].extend(body[key])
                comb_data_1[key].extend(hand[key])
        else:
            comb_data_2[key] = []
            if isinstance(body[key][0], (list, np.ndarray)):
                for i in range(len(body[key])):
                    comb_data_2[key].extend(body[key][i])
                    comb_data_2[key].extend(np.ravel(hand[key][i]))
            else:
                comb_data_2[key].extend(body[key])
                comb_data_2[key].extend(hand[key])
                
    return comb_data_1, comb_data_2

# Appends hand values at the end of the each frame, but unlike combine_body_and_hand, does not flatten the final output. Final 2-d output shape = (60 videos, 75 frames).
def get_frame_by_frame(body, hand):
    
    frame_by_frame_1 = {}; frame_by_frame_2 = {}
    for key in body:
       if key.split('_')[0][-1] == '1':
           frame_by_frame_1[key] = []
           for i in range(len(body[key])):
               temp = []
               temp.extend(body[key][i])
               temp.extend(np.ravel(hand[key][i]))
               frame_by_frame_1[key].append(temp)
       else:
           frame_by_frame_2[key] = []
           for i in range(len(body[key])):
               temp = []
               temp.extend(body[key][i][0:2])
               temp.extend(np.ravel(hand[key][i]))
               frame_by_frame_2[key].append(temp)
    
    return frame_by_frame_1, frame_by_frame_2

# Converts (x,y,w,h) hand box to set of 4 corner coordinates
def convert_hand_to_coord(hand):
    coords = []
    coords.append(hand[0]); coords.append(hand[1]) #URH
    coords.append(hand[0]); coords.append(hand[1]+hand[3]) #LRH
    coords.append(hand[0]+hand[2]); coords.append(hand[1]) #ULH
    coords.append(hand[0]+hand[2]); coords.append(hand[1]+hand[3]) #LLH
    return coords

# Averages hand box coordinates across 75 frames. Ignores hand boxes which are (0,0,0,0) which represent an undetected hand. Each frame contains 8 hand box coordinates. Returns 8 averaged (x,y) coordinates)
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
            avgd[i] = 0 #nanchange
        else:
            avgd[i] = np.mean(avgd[i])

    # separates into corner points (x,y)
    #avgd = [avgd[i:i+2] for i in range(0, len(avgd), 2)]
    
    return avgd
    
# Averages body coordinates across 75 frames. Ignores body points which are (0,0) which represent an undetected point. Each frame contains 18 body coordinates. Returns 18 averaged (x,y) coordinates)
def average_body(video):
    
    avgd = [[] for i in range(len(video[0]))]
    for frame in video:
        for i in range(0, len(frame), 2):
            if int(frame[i]) == 0 and int(frame[i+1]) == 0: # change-1
                continue
            avgd[i].append(frame[i]); avgd[i+1].append(frame[i+1])
    for i in range(len(avgd)):
        if len(avgd[i]) == 0:
            avgd[i] = 0 #nanchang
        else:
            avgd[i] = np.mean(avgd[i])
    
    # separates into corner points (x,y)
    #avgd = [avgd[i:i+2] for i in range(0, len(avgd), 2)]

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
            if (i%4) == 0:
                if int(val) == -1 and int(frame[i+1]) == -1:
                    clnd[j].append(0)
                    continue
            if (i%4) == 1:
                if int(val) == -1 and int(frame[i-1]) == -1:
                    clnd[j].append(0)
                    continue
            clnd[j].append(val)
    return clnd

# Returns a vector of all pairs that do not include the video index in a dissimilarity matrix, and a vector of all pairs that do include the video index.
def hold_pairs(hold, matrix, removed_videos):
    held_values = []
    vector = []
    # Moves row by row through top triangle of distance matrix. Rows get shorter and shorter as loop travels down the triangle.
    for i in range(len(matrix)):
        if i in removed_videos:
            continue
        for j in range(i+1,len(matrix)):
            if j in removed_videos:
                continue
            if i is hold or j is hold:
                held_values.append(matrix[i][j])
                continue
            vector.append(matrix[i][j])
    return np.array(vector), np.array(held_values)

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
    #df = df.astype(int)
    #df = df.apply(zscore)
    
    return df, actions
    
# Plot and save dendrogram produced from hierarchical clustering analysis (clustering method provided in argument). Returns hierarchical clustering solution as a list.
def plot_heir_cluster(df, met, dir, vid_set_num):
    link = sch.linkage(df, method=met)
    dg = sch.dendrogram(link, orientation="right", leaf_font_size=6, labels=df.index)
    plt.title('Video Set {}'.format(vid_set_num))
    plt.xlabel('Videos')
    plt.gcf().set_size_inches((10,10))
    plt.subplots_adjust(left=0.22)
    #plt.savefig('{}/{}_vidset{}_dgram.png'.format(dir, met, vid_set_num))
    plt.show()
    plt.close()
    return dg['leaves']

def sum_two_matrices(m1, m2):

    new_matrix = np.empty(m1.shape)
    
    for i in range(m1.shape[0]):
        for k in range(m2.shape[0]):
            if np.isnan(m1[i][k]):
                if np.isnan(m2[i][k]): # both nans
                    new_matrix[i][k] = np.nan
                else: # only m1[i][k] is nan
                    new_matrix[i][k] = m2[i][k]
            elif np.isnan(m2[i][k]): # only m2[i][k] is nan
                new_matrix[i][k] = m1[i][k]
            else: # none are nan
                new_matrix[i][k] = m1[i][k] + m2[i][k]
        
    return new_matrix

def plot_heatmap(df, index, columns, values, title=None):
    
    set_trace()

    ax = plt.axes()
    df = df.pivot(index=index, columns=columns, values=values)
    df = df.astype(float)
    sns.heatmap(df, xticklabels=True, yticklabels=True)
    if title:
        ax.set_title(title)
    plt.show()

def plot_histograms(df, joint, video, title=None):
    
    ax = plt.axes()
    videos = df[video].unique()
    joints = df[joint].unique()
    
    vid_values = {v: 0 for v in videos}
    joint_values = {j: 0 for j in joints}
    
    for i, row in df.iterrows():
        if row['frequency'] > 0:
            vid_values[row['video']] += 1
            joint_values[row['joint']] += 1
    
    v_values = [[vid_values[v], v] for v in vid_values]
    j_values = [[joint_values[j], j] for j in joint_values]
    
    v_values.sort(); j_values.sort()
    
    set_trace()
    
    return
    
    # Count number of joints for each video
    
    
    
# Plot and save correlation heatmap of dataframe
#def plot_heatmap(df, met, dir, vid_set_num):
#    corr = df.corr()
#    fig, ax = plt.subplots(figsize=(18, 15))
#    colormap = sns.diverging_palette(220, 10, as_cmap=True)
#    sns.heatmap(corr, cmap=colormap)
#    plt.title('Video Set {}'.format(vid_set_num), fontsize=20)
#    plt.subplots_adjust(left=0.22, bottom=0.20, top=0.95, right=0.90)
#    plt.xticks(range(len(corr.columns)), corr.columns);
#    plt.yticks(range(len(corr.columns)), corr.columns)
#    plt.savefig('{}/{}_vidset{}_heatmap.png'.format(dir, met, vid_set_num))
#    #plt.show()
#    plt.close()
#    return

# Randomly shuffles dataset and correlates first half of shuffled set to second half of shuffled set. Repeats 'num' times. Used to estimate reliability of dataset.

def get_split_half_reliabilities(num, data):
    
    corr_sum = 0
    for i in range(num):
        shuffled = random.sample(list(data), len(data))
        mid = int(len(data)/2)
        half1 = shuffled[:mid]; half2 = shuffled[mid:2*mid]
        half1 = np.array(half1, dtype='float').flatten()
        half2 = np.array(half2, dtype='float').flatten()
        corr_sum += pearsonr(half1, half2)[0]
    return corr_sum/num
    
    '''
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
    '''

def plot_layer_fit(reg, title):
    
    fig, ax = plt.subplots()
    #labels = list(scores.keys())
    #labels.insert(0, '')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.axhline(y=0, color='black')
    ax.axhline(y=0, color='black')

    set_trace()

    colors = ['black', 'blue', 'red', 'green']

    #means = np.array([np.nanmean(reg[layer]) for layer in reg][:-1])
    #k = np.arange(len(means))
    
    for i, r in enumerate(reg):
        m = ax.plot(np.arange(len(reg[r])), reg[r], marker='.', c=colors[i], linestyle='-', linewidth=2, label=r)

#    fit = np.polyfit(np.ravel(np.argwhere(~np.isnan(means))), means[~np.isnan(means)], deg)
#    fit_fn = np.poly1d(fit)
#
#    f = ax.plot(k, fit_fn(k), marker='.', c='blue', linestyle='-', linewidth=2, zorder=1, label=str(fit))
    
    ax.legend(loc ='upper right')
    plt.show()

def plot_layer_bar(scores, labels, title, noise_ceiling=None):

    set_trace()

    fig, ax = plt.subplots()
    #labels = list(scores.keys())
    #labels.insert(0, '')
    ax.set_ylabel('Scores')
    ax.set_xlabel('Layers')
    ax.set_title(title)
    ax.axhline(y=0, color='black')
    ax.axhline(y=0, color='black')
    ax.set_ylim(bottom=-0.015, top=0.20)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, fontsize=7)
    #x_pos = 1

    colors = ['purple', 'blue', 'green', 'red']
    stream_starts = [1, 16, 21, 28, 35, 42, 49, 55]
    stream_ys = [-0.005 for i in range(len(stream_starts))]
    #i = 0
    
    for i, behavior in enumerate(scores):
        means = [np.nanmean(layer) for layer in scores[behavior]][:-1]
        ax.plot(range(1,56), means, color=colors[i], ls='-', label=behavior)
        
    for i in range(len(stream_starts)-1):
#        if i > 1:
#            continue
        height = -0.005
        center = (stream_starts[i+1]+stream_starts[i])/2
        width = stream_starts[i+1]-stream_starts[i]
        ax.annotate(i, xy=(center,0.185), xytext=(center,0.185), xycoords='data', fontsize=12, ha='center', va='bottom', bbox=dict(boxstyle='square', fc='white'), arrowprops=dict(arrowstyle='-[, widthB={}, lengthB=0.75'.format(width/2.1), lw=1))
    
    #ax.scatter(stream_starts, stream_ys, color='black', marker='*', label='Start of Stream')
        
    ax.legend(loc='lower right')
    plt.show()
    
#    for layer in scores:
#
#        s = scores[layer]
#        if int(layer) in stream_starts:
#            i = 1
#
#        mean = np.nanmean(s)
#        bar_middle = 0.5*(np.nanpercentile(s,25) + np.nanpercentile(s,75))
#        bar_height = np.nanpercentile(s,75) - np.nanpercentile(s,25)
#        median = np.nanpercentile(s,50)
#        mins = np.nanmin(s); maxs = np.nanmax(s)
#
#        ax.barh(y=bar_middle, height=bar_height, width=0.5, left=x_pos-0.25, color='white', edgecolor=colors[i])
#        ax.plot([x_pos-0.245, x_pos+0.245], [median, median], marker=None, c=colors[i], linewidth=1)
#        ax.plot([x_pos, x_pos], [mins, maxs], marker=None, c=colors[i], linestyle='dashed', linewidth=1, zorder=0)
#        ax.plot([x_pos-.12, x_pos+.12], [mins, mins], marker=None, c=colors[i], linewidth=1)
#        ax.plot([x_pos-.12, x_pos+.12], [maxs, maxs], marker=None, c=colors[i], linewidth=1)
#        ax.plot([x_pos-.245, x_pos+.245], [mean, mean], c='red', marker=None, linewidth=1)
#
#        if wilcoxon(s, alternative='greater')[1] < 0.01:
#            ax.plot(x_pos, -0.7, marker='*', color='black')
#
#        x_pos += 1
#        i = 0
#
#    if noise_ceiling:
#        nc = noise_ceilings[behavior]
#        noise_middle = 0.5*(nc[0]+nc[1])
#        noise_height = nc[1]-nc[0]
#        ax.barh(y = noise_middle, height=noise_height, width=0.5, left=x_pos-0.25, color='lightgray', zorder=0)
#
#    plt.show()


def plot_violin_plot(scores, behaviors, noise_ceilings=None, title=None):

    set_trace()

    colors = [(.15,.25,.75), (.15,.5,.15), (.75,.25,.25)]
    models = {'avg': 'Averaged Poses', 'pro': 'Procrustes', 'parts': 'Body Parts', 'avg-cent': 'Centroids'}
    
    fig, axs = plt.subplots(1, len(behaviors))
    fig.text(0.45, 0.025, 'Prediction Model', fontsize=10, fontweight='bold')
    fig.text(0.05, 0.35, 'Prediction Performance', rotation='vertical', fontsize=10, fontweight='bold')
    if title:
        fig.suptitle(title)
    
    for i, behavior in enumerate(scores[list(scores.keys())[0]]):
        
        if len(behaviors) == 1:
            ax = axs
        else:
            ax = axs[i]
        
        s = []
        for model in scores:
            for score in scores[model][behavior]:
                s.append([models[model], score])
                
        set_trace()
        
        df = pd.DataFrame(s, columns=['Model', 'Score'])
    
        sns.violinplot(ax=ax, x='Model', y='Score', data=df, cut=0, color=colors[i])
        ax.set_title(behavior)
        ax.set_ylim(bottom=-1, top=1)
        ax.set_xlim(left=-0.5, right=len(scores)-0.5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        if noise_ceilings:
            nc = noise_ceilings[behavior]
            noise_middle = 0.5*(nc[0]+nc[1])
            noise_height = nc[1]-nc[0]
            ax.barh(y = noise_middle, height=noise_height, width=4, left=-1, color=(.22, .22, .22, .5), zorder=3)
        
    
    plt.rc('xtick',labelsize=5)
    plt.rc('ytick',labelsize=5)
    plt.show()

# Plots error bar given the data set of prediction accuracy scores from cross-validation.
def plot_error_bar(scores, noise_ceilings=None, title=None):
    
    set_trace()
    
    fig, ax = plt.subplots()
    labels = ['','']
    models = {'avg': 'Averaged Poses', 'pro': 'Procrustes', 'parts': 'Body Parts Involved'}
    
#    set_trace()
    
#    for behavior in scores[list(scores.keys())[0]]:
#        for model in scores:
#            #labels.insert(-1,'{}-{}'.format(model, behavior))
#            labels.insert(-1, behavior)
    ax.set_ylabel('Prediction Performance', fontsize=10, fontweight='bold')
    ax.set_xlabel('Behavior Being Predicted', fontsize=10, fontweight='bold')
    rcParams['axes.labelpad'] = 30
    ax.set_title(title)
    ax.axhline(y=0, color='black')
    ax.axhline(y=0, color='black')
#    ax.set_xticklabels(labels)
    x_pos = 1
    
    t_colors = [(.15,.25,.75,.5), (.15,.5,.15,.5), (.75,.25,.25,.5)]
    colors = [(.15,.25,.75), (.15,.5,.15), (.75,.25,.25)]
    
    for i, model in enumerate(scores):
        
        means = []; bar_middles = []; bar_heights = []; medians = []; xs = []
        
        for j, behavior in enumerate(scores[model]):
        
            labels.insert(-1, behavior)
        
            s = scores[model][behavior]
            
            mean = np.mean(s)
            bar_middle = 0.5*(np.percentile(s,25) + np.percentile(s,75))
            bar_height = np.percentile(s,75) - np.percentile(s,25)
            median = np.percentile(s,50)
            x = x_pos + j*len(scores)
           
            means.append(mean)
            bar_middles.append(bar_middle)
            bar_heights.append(bar_height)
            medians.append(median)
            xs.append(x)
            
            ax.plot([x-0.245, x+0.245], [median, median], marker=None, c='black', linewidth=1)
            ax.plot([x, x], [min(s), max(s)], marker=None, c=colors[i], linestyle='dashed', linewidth=1, zorder=0)
            ax.plot([x-.12, x+.12], [min(s), min(s)], marker=None, c=colors[i], linewidth=1)
            ax.plot([x-.12, x+.12], [max(s), max(s)], marker=None, c=colors[i], linewidth=1)
            ax.plot([x-.245, x+.245], [mean, mean], c='red', marker=None, linewidth=1)
            
            if noise_ceilings:
                nc = noise_ceilings[behavior]
                noise_middle = 0.5*(nc[0]+nc[1])
                noise_height = nc[1]-nc[0]
                ax.barh(y = noise_middle, height=noise_height, width=0.7, left=x-0.35, color=(.22, .22, .22, .5), zorder=3)
        
        xs = np.array(xs)
        ax.barh(y=bar_middles, height=bar_heights, width=0.5, left=xs-0.25, color=t_colors[i], edgecolor='black', label=models[model])

        x_pos += 1
    
    patch = mpatches.Patch(color=(.22, .22, .22, .5), label='Noise Ceiling')
    line = Line2D([0], [0], color='red', linewidth=1, linestyle='-', label='Mean')
    
    handles, l = ax.get_legend_handles_labels()
    handles.extend([patch, line])
    
    ax.legend(handles=handles, loc='best')
    ax.set_xticklabels(labels)
    ax.set_xticks(np.arange(len(labels)))
    
    plt.show()
        
    set_trace()
    
    for behavior in scores[list(scores.keys())[0]]:
        for i, model in enumerate(scores):
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
            
            if noise_ceilings:
                nc = noise_ceilings[behavior]
                noise_middle = 0.5*(nc[0]+nc[1])
                noise_height = nc[1]-nc[0]
                ax.barh(y = noise_middle, height=noise_height, width=0.7, left=x_pos-0.35, color=(.22, .22, .22, .5), zorder=3)
        
            x_pos += 1
    
    ax.set_xticks(np.arange(len(labels)))
    
    plt.show()
