import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

from pdb import set_trace

# columns = vids, index = points
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
            #b_values[i].append(val)
            c_box_values[i].append(val)
            c_pose_values[i].append(val)
        for i, val in enumerate((h_box_data[key])):
            #h_values[i].append(val)
            #c_values[b_len+i].append(val)
            c_box_values[b_len+i].append(val)
        for i, val in enumerate((h_pose_data[key])):
            c_pose_values[b_len+i].append(val)
    
#    b_df = pd.DataFrame(b_values, columns=actions)
#    h_df = pd.DataFrame(h_values, columns=actions)
    c_box_df = pd.DataFrame(c_box_values, columns=actions)
    c_pose_df = pd.DataFrame(c_pose_values, columns=actions)
    return c_box_df, c_pose_df

'''
def to_df(data, names):
    values = [[] for i in range(len(data[list(data.keys())[0]]))]
    actions = []
    for key in data:
        i = names[names['Video name']==key+'.mp4']
        if i.empty:
            continue
        actions.append(i.iloc[0]['Action Name'])
        for i, val in enumerate(data[key]):
            values[i].append(val)
    df = pd.DataFrame(values, columns=actions)
    return df
'''

#'complete', 'average', 'weighted', 'ward'
#handpose, ward, handonly
#[22, 59, 30, 5, 7, 56, 18, 58, 35, 28, 33, 39, 19, 55, 40, 20, 9, 37, 31, 50, 49, 48, 45, 43, 42, 38, 34, 27, 26, 23, 21, 12, 10, 8, 1, 2, 17, 29, 24, 0, 32, 51, 57, 52, 53, 3, 15, 13, 41, 46, 54, 4, 36, 14, 47, 16, 11, 44, 6, 25]
#15000, __, 16000
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

# Creates heatmap of datagram
def plot_heatmap(df, met, h_data_set, vid_set_num):
    #set_trace()
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    sns.heatmap(corr, cmap=colormap)
    plt.title('Video Set {}'.format(vid_set_num), fontsize=20)
    #sns.set(font_scale=0.6)
    plt.subplots_adjust(left=0.22, bottom=0.20, top=0.95, right=0.90)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('plots/{}_{}_{}_heatmap.png'.format(met, h_data_set, vid_set_num))
    #plt.show()
    plt.close()
    return

if __name__ == "__main__":
    
    methods = ['complete', 'average', 'weighted', 'ward']
    #methods = ['average']
    body = scipy.io.loadmat('vids_set_body.mat')
    hand_box = scipy.io.loadmat('vids_set_hand.mat')
    hand_pose = scipy.io.loadmat('set1_hand.mat')
    xls = pd.ExcelFile('vidnamekey.xlsx')
    
    # Creates df of action names
    set1_names =  xls.parse(xls.sheet_names[0])
    set2_names =  xls.parse(xls.sheet_names[1])
    names = pd.concat([set1_names, set2_names])
    
    del body['__header__']; del hand_box['__header__']; del hand_pose['__header__']
    del body['__version__']; del hand_box['__version__']; del hand_pose['__version__']
    del body['__globals__']; del hand_box['__globals__']; del hand_pose['__globals__']
    
    # Unravels each 2d array in body and hand
    for key in body:
        body[key] = np.ravel(body[key])
        hand_box[key] = np.ravel(hand_box[key])
        hand_pose[key] = np.ravel(hand_pose[key])

    # Creates DataFrames
    body_and_box_df_1, body_and_pose_df_1 = to_df(body, hand_box, hand_pose, set1_names)
    body_and_box_df_2, body_and_pose_df_2 = to_df(body, hand_box, hand_pose, set2_names)
    dfs = [[body_and_box_df_1, body_and_box_df_2, 'box'], [body_and_pose_df_1, body_and_pose_df_2, 'pose']]
    
    for m in methods:
        for df in dfs: # [hand data set, df for set 1, df for set 2]
            #hierarchy_1 =
            for i in range(2):
                order = plot_heir_cluster(df[i].T, m, df[2], i+1)
                hierarchy = [list(df[i].columns)[j] for j in order]
                sorted = df[i].reindex(columns=hierarchy)
                plot_heatmap(sorted, m, df[2], i+1)
            
