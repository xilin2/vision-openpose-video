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

def new_to_df(b_data, h_data, names):

    b_values = [[] for i in range(len(b_data[list(b_data.keys())[0]]))]
    h_values = [[] for i in range(len(h_data[list(h_data.keys())[0]]))]
    c_values = [[] for i in range(len(b_values)+len(h_values))]
    b_len = len(b_values)
    actions = []
    
    for key in b_data:
        #set_trace()
        i = names[names['Video name']==key+'.mp4']
        if i.empty:
            continue
        actions.append(i.iloc[0]['Action Name'])
        for i, val in enumerate(b_data[key]):
            b_values[i].append(val)
            c_values[i].append(val)
        for i, val in enumerate(h_data[key]):
            h_values[i].append(val)
            c_values[b_len+i].append(val)
    
    b_df = pd.DataFrame(b_values, columns=actions)
    h_df = pd.DataFrame(h_values, columns=actions)
    c_df = pd.DataFrame(c_values, columns=actions)
    return b_df, h_df, c_df

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

#15000, __, 16000
def plot_heir_cluster(df, label):
    while True:
        set_trace()
        dg = sch.dendrogram(sch.linkage(df, method  = "ward"), leaf_rotation=90, leaf_font_size=5, labels=df.index)
        plt.title(label)
        plt.xlabel('Videos')
        #plt.ylabel('Euclidean distances')
        plt.show()

# Creates heatmap of datagram
def plot_heatmap(df, label):
    #Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(60, 60))
    #Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    #sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    sns.heatmap(corr, cmap=colormap)
    sns.set(font_scale=0.2)
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    # plt.savefig(label+'_heatmap.png')
    plt.show()
    # plt.close()
    return

if __name__ == "__main__":
    
    body = scipy.io.loadmat('vids_set_body.mat')
    hand = scipy.io.loadmat('vids_set_hand.mat')
    xls = pd.ExcelFile('vidnamekey.xlsx')
    
    # Creates df of action names
    set1_names =  xls.parse(xls.sheet_names[0])
    set2_names =  xls.parse(xls.sheet_names[1])
    names = pd.concat([set1_names, set2_names])
    
    del body['__header__']; del hand['__header__']
    del body['__version__']; del hand['__version__']
    del body['__globals__']; del hand['__globals__']
    
    # Unravels each 2d array in body and hand
    for key in body:
        body[key] = np.ravel(body[key])
        hand[key] = np.ravel(hand[key])

    # Creates DataFrames
    body_df = to_df(body, set1_names)
    hand_df = to_df(hand, set1_names)
    b_df, h_df, c_df = new_to_df(body, hand, set1_names)
    
    #plot_heir_cluster(b_df.T, 'body')
    #plot_heir_cluster(h_df.T, 'hand')
    plot_heir_cluster(c_df.T, 'combined')

    set_trace()
    # Creates heatmap
    plot_heatmap(b_df, 'body')
    plot_heatmap(h_df, 'hand')
    plot_heatmap(c_df, 'combined')
