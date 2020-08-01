import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import correlate
from scipy.stats import zscore

from pdb import set_trace

# Return two dataframes with body + hand box features, and body + hand pose features of index length 6000 and 14850, respectively. The index corresponds to a value within each video's dataset, and the columns correspond to the video list.
def to_df(b_data, h_box_data, h_pose_data, names):

    b_len = len(b_data[list(b_data.keys())[0]])
    h_box_len = len(h_box_data[list(h_box_data.keys())[0]])
    h_pose_len = len(h_pose_data[list(h_pose_data.keys())[0]])
    c_box_values = [[] for i in range(b_len+h_box_len)]
    c_pose_values = [[] for i in range(b_len+h_pose_len)]
    actions = []
    
    for key in b_data:
        i = names[names['Video name']==key+'.mp4']
        if i.empty:
            continue
        actions.append(i.iloc[0]['Action Name'])
        for i, val in enumerate(b_data[key]):
            c_box_values[i].append(val)
            c_pose_values[i].append(val)
        for i, val in enumerate((h_box_data[key])):
            c_box_values[b_len+i].append(val)
        for i, val in enumerate((h_pose_data[key])):
            c_pose_values[b_len+i].append(val)

    c_box_df = pd.DataFrame(c_box_values, columns=actions)
    c_pose_df = pd.DataFrame(c_pose_values, columns=actions)
    c_box_df = c_box_df.apply(zscore)
    c_pose_df = c_pose_df.apply(zscore)

    return c_box_df, c_pose_df
    
#15000, __, 16000
# Plot and save dendrogram produced from hierarchical clustering analysis (clustering method provided in argument). Returns hierarchical clustering solution as a list.
def plot_heir_cluster(df, met, h_data_set, vid_set_num):
    link = sch.linkage(df, method=met)
    dg = sch.dendrogram(link, orientation="right", leaf_font_size=6, labels=df.index)
    plt.title('Video Set {}'.format(vid_set_num))
    plt.xlabel('Videos')
    plt.gcf().set_size_inches((10,10))
    plt.subplots_adjust(left=0.22)
    plt.savefig('plots/{}_{}_{}_dgram.png'.format(met, h_data_set, vid_set_num))
    #plt.show()
    plt.close()
    return dg['leaves']

# Plot and save correlation heatmap of dataframe
def plot_heatmap(df, met, h_data_set, vid_set_num):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap)
    plt.title('Video Set {}'.format(vid_set_num), fontsize=20)
    plt.subplots_adjust(left=0.22, bottom=0.20, top=0.95, right=0.90)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('plots/{}_{}_{}_heatmap.png'.format(met, h_data_set, vid_set_num))
    #plt.show()
    plt.close()
    return

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_features = dir_path + '/Yolo-Hand-Detection/exp/vids_set/'
    
    # Load name and feature data from files into dictionary
    body = scipy.io.loadmat(dir_features + 'vids_set_body.mat')
    hand_box = scipy.io.loadmat(dir_features + 'vids_set_hand.mat')
    hand_pose = scipy.io.loadmat(dir_features + 'vids_set_fingers.mat')
    xls = pd.ExcelFile('vidnamekey.xlsx')

    # Delete extra dictionary keys
    del body['__header__']; del hand_box['__header__']; del hand_pose['__header__']
    del body['__version__']; del hand_box['__version__']; del hand_pose['__version__']
    del body['__globals__']; del hand_box['__globals__']; del hand_pose['__globals__']
    
    # Create DataFrame of action names
    set1_names =  xls.parse(xls.sheet_names[0])
    set2_names =  xls.parse(xls.sheet_names[1])
    #names = pd.concat([set1_names, set2_names])
    
    # Unravel the 2d array stored under each body and hand dictionary key
    for key in body:
        body[key] = np.ravel(body[key])
        hand_box[key] = np.ravel(hand_box[key])
        hand_pose[key] = np.ravel(hand_pose[key])

    # Create 4 DataFrames corresponding to different video sets and hand data sets
    body_and_box_df_1, body_and_pose_df_1 = to_df(body, hand_box, hand_pose, set1_names)
    body_and_box_df_2, body_and_pose_df_2 = to_df(body, hand_box, hand_pose, set2_names)
    dfs = [[body_and_box_df_1, body_and_box_df_2, 'box'], [body_and_pose_df_1, body_and_pose_df_2, 'pose']]
    
    methods = ['complete', 'average', 'weighted', 'ward']
    
    for df in dfs: # df = [box or pose, df for set 1, df for set 2]
        cl = correlate(df[0].corr(), df[1].corr()) # Calculate correlation matrix between video sets 1 and 2's dissimilarity matrices
        print('Correlation for body pose and hand {} features between set 1 and 2 = {}'.format(df[2], cl.mean()))
        for m in methods:
            for i in range(2):
                order = plot_heir_cluster(df[i].T, m, df[2], i+1)
                hierarchy = [list(df[i].columns)[j] for j in order]
                sorted = df[i].reindex(columns=hierarchy) # Arrange DF according to hierarchical clustering solution
                plot_heatmap(sorted, m, df[2], i+1) # Plot correlation heatmap of sorted DF

