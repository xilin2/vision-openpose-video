import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
import os
import random

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import correlate
from scipy.stats import zscore, kendalltau, pearsonr

from pdb import set_trace

# Return unraveled dataframes with body + hand features. The index corresponds to a value within each video's dataset, and the columns correspond to the videos in the video set.
def to_df(data, names):
    
    # Unravel the 2d array stored under each body and hand dictionary key
    unraveled = {}
    for key in data:
        temp = []
        for frame in data[key]:
            for el in frame:
                temp.append(el)
        unraveled[key] = temp
    
    actions = []
    values = [[] for i in range(len(unraveled[list(unraveled.keys())[0]]))]
    
    for key in unraveled:
        i = names[names['Video name']==key+'.mp4']
        if i.empty:
            continue
        actions.append(i.iloc[0]['Action Name'])
        for i, val in enumerate(unraveled[key]):
            values[i].append(val)
    
    df = pd.DataFrame(values, columns=actions)
    df = df.apply(zscore)
    
    return df
    
    '''
    b_len = len(b_data[list(b_data.keys())[0]])
    h_box_len = len(h_box_data[list(h_box_data.keys())[0]])
    #h_pose_len = len(h_pose_data[list(h_pose_data.keys())[0]])
    c_box_values = [[] for i in range(b_len+h_box_len)]
    #c_pose_values = [[] for i in range(b_len+h_pose_len)]
    actions = []
    
    for key in b_data:
        i = names[names['Video name']==key+'.mp4']
        if i.empty:
            continue
        actions.append(i.iloc[0]['Action Name'])
        for i, val in enumerate(b_data[key]):
            c_box_values[i].append(val)
            #c_pose_values[i].append(val)
        for i, val in enumerate((h_box_data[key])):
            c_box_values[b_len+i].append(val)
#        for i, val in enumerate((h_pose_data[key])):
#            c_pose_values[b_len+i].append(val)
            
    c_box_df = pd.DataFrame(c_box_values, columns=actions)
    #c_pose_df = pd.DataFrame(c_pose_values, columns=actions)
    c_box_df = c_box_df.apply(zscore)
    #c_pose_df = c_pose_df.apply(zscore)
    
    return c_box_df
    '''
    
#15000, __, 16000
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

if __name__ == "__main__":
    
    # Load name and feature data from files into dictionary
    label = 'empty'
    methods = ['complete', 'average', 'weighted', 'ward']
#    body_og = scipy.io.loadmat('vids_set_body.mat')
#    hand_box_og = scipy.io.loadmat('vids_set_hand.mat')
    body = scipy.io.loadmat('vids_body_new_track.mat')
    hand = scipy.io.loadmat('vids_hand_new_track.mat')
#    hand_pose = scipy.io.loadmat('set1_hand.mat')
    xls = pd.ExcelFile('vidnamekey.xlsx')
    
    dir = '{}_plots'.format(label)
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    # Delete extra dictionary keys
    del body['__header__']; del hand['__header__']
    del body['__version__']; del hand['__version__']
    del body['__globals__']; del hand['__globals__']
    
    # Create DataFrame of action names
    set1_names =  xls.parse(xls.sheet_names[0])
    set2_names =  xls.parse(xls.sheet_names[1])
    names = pd.concat([set1_names, set2_names])
    
    # Combine body and hand dictionaries
    comb_data_1 = {}; comb_data_2 = {}
    for key in body:
        if key.split('_')[0][-1] == '1':
            comb_data_1[key] = body[key].tolist()
            for i in range(75):
                comb_data_1[key][i].extend(hand[key][i].tolist())
        else:
            comb_data_2[key] = body[key].tolist()
            for i in range(75):
                comb_data_2[key][i].extend(hand[key][i].tolist())
    
    # Create 4 DataFrames corresponding to different video sets and hand data sets
    body_and_box_df_1 = to_df(comb_data_1, set1_names)
    body_and_box_df_2 = to_df(comb_data_2, set2_names)
    dfs = [[body_and_box_df_1, body_and_box_df_2, 'handbox']]
    
    # Reliabilities
    reliabilities = get_split_half_reliabilities(1000, comb_data_1)
    for el in sorted(reliabilities.items(), key=lambda x:x[1]):
        print(el)
    values_only = [reliabilities[i] for i in reliabilities]
    set_trace()
    print('Reliability mean: {}'.format(np.mean(values_only)))
    print('Reliability median: {}'.format(np.median(values_only)))

    # Analysis requiring unraveled frame data
    for df in dfs: # df = ['handbox' or 'handpose', df for set 1, df for set 2]
        
        cl = correlate(df[0].corr(), df[1].corr()) # Calculate correlation matrix between video sets 1 and 2's dissimilarity matrices
        # n = df[0].corrwith(df[1], axis=0)
        
        #print('Kendall\'s tau for body pose and {} features between set 1 and 2 = {}'.format(df[2], kendalltau(arr_set_1, arr_set_2)[0]))
        print('Correlation for body pose and {} features between set 1 and 2 = {}'.format(df[2], cl.mean()))
        
        '''
        for m in methods:
            for i in range(2):
                order = plot_heir_cluster(df[i].T, m, dir, i+1)
                hierarchy = [list(df[i].columns)[j] for j in order]
                sorted = df[i].reindex(columns=hierarchy) # Arrange DF according to hierarchical clustering solution
                plot_heatmap(sorted, m, dir, i+1) # Plot correlation heatmap of sorted DF
        '''
