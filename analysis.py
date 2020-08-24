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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from pdb import set_trace

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

def convert_hand_to_coord(hand):
    coords = []
    coords.append(hand[0]); coords.append(hand[1]) #URH
    coords.append(hand[0]); coords.append(hand[1]+hand[3]) #LRH
    coords.append(hand[0]+hand[2]); coords.append(hand[1]) #ULH
    coords.append(hand[0]+hand[2]); coords.append(hand[1]+hand[3]) #LLH
    return coords
    
def average_hands(video):

    avgd = [[] for i in range(16)]
    for frame in video:
        separated_by_hand = [frame[i:i+4] for i in range(0, 8, 4)]
        for i, hand in enumerate(separated_by_hand):
            if all(v == 0 for v in hand):
                continue
            hand = convert_hand_to_coord(hand)
            for j, val in enumerate(hand):
                avgd[j+8*i].append(val)
    for i in range(len(avgd)):
        if len(avgd[i]) == 0:
            avgd[i] = 0
        else:
            avgd[i] = np.mean(avgd[i])
    
    return(avgd)
    
# Returns averaged body frame
def average_body(video):

    avgd = [[] for i in range(17*2)]
    for frame in video:
        rmvd_score_id = []
        for i, val in enumerate(frame):
            if (i%4) == 3 or (i%4) == 2:
                continue
            rmvd_score_id.append(val)
        for i in range(0, len(avgd), 2):
            point = (rmvd_score_id[i], rmvd_score_id[i+1])
            if int(point[0]) == 0 and int(point[1]) == 0:
                continue
            avgd[i].append(point[0]); avgd[i+1].append(point[1])
    for i in range(len(avgd)):
        if len(avgd[i]) == 0:
            avgd[i] = 0
        else:
            avgd[i] = np.mean(avgd[i])

    return avgd

def hold_pairs(hold, matrix):
    held_values = []
    vector = []
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            if i is hold or j is hold:
                held_values.append([matrix[i][j]])
                continue
            vector.append([matrix[i][j]])
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
    
    average_data = True; original_data = False
    use_PCA = False
    reg_regression = True
    label = 'empty'
    methods = ['complete', 'average', 'weighted', 'ward']
    
    # Load feature data from files into dictionary
#    body_og = scipy.io.loadmat('vids_set_body.mat')
#    hand_box_og = scipy.io.loadmat('vids_set_hand.mat')
    body = scipy.io.loadmat('vids_body_new_track.mat')
    hand = scipy.io.loadmat('vids_hand_new_track.mat')
    
    # Load name data
    xls = pd.ExcelFile('vidnamekey.xlsx')
    
    # Load behavioral data
    goals = scipy.io.loadmat('data/GoalSimilarity.mat')['GuidedBehaviorModel'][0][0][1]
    intuitive_actions = scipy.io.loadmat('data/IntuitiveActionSim.mat')['IntuitiveSimilarity'][0][0][1][0][0][1]
    movement = scipy.io.loadmat('data/MovementSimilarity.mat')['GuidedBehaviorModel'][0][0][1]
    visual = scipy.io.loadmat('data/VisualSimilarity.mat')['GuidedBehaviorModel'][0][0][1]
    
    behavioral_data = {'Goals': goals, 'Intuitive Actions': intuitive_actions, 'Movement': movement, 'Visual': visual}
    
        
    # Delete extra dictionary keys
    del body['__header__']; del hand['__header__']
    del body['__version__']; del hand['__version__']
    del body['__globals__']; del hand['__globals__']
    
    # Create DataFrame of action names
    set1_names =  xls.parse(xls.sheet_names[0])
    set2_names =  xls.parse(xls.sheet_names[1])
    names = pd.concat([set1_names, set2_names])
    
    ''' Data organization '''
    
    orgd_body = {}; orgd_hand = {}

    # Unravel complete data
    if original_data:
        for key in body:
            orgd_body[key] = []; orgd_hand[key] = []
            for frame_index in range(len(body[key])):
                for el in body[key][frame_index]:
                    orgd_body[key].append(el)
                for el in hand[key][frame_index]:
                    orgd_hand[key].append(el)
    
    # Average data
    elif average_data:
        avgd_body = {}; avgd_hand = {}
        for key in body:
            orgd_body[key] = average_body(body[key])
            orgd_hand[key] = average_hands(hand[key])
    
    # Combine body and hand dictionaries
    comb_data_1, comb_data_2 = combine_body_hand(orgd_body, orgd_hand)
    
    # Reliabilities
    '''
    print('Getting reliabilities')
    reliabilities = get_split_half_reliabilities(1000, comb_data_1)
#    for el in sorted(reliabilities.items(), key=lambda x:x[1]):
#        print(el)
    values_only = [reliabilities[i] for i in reliabilities]
    print('Reliability mean: {}'.format(np.mean(values_only)))
    print('Reliability median: {}'.format(np.median(values_only)))
    '''
    
    # Create DataFrames corresponding to different video sets and hand data sets
    #print('Creating dataframes')
    body_and_box_df_1, actions_1 = to_df(comb_data_1, set1_names)
    body_and_box_df_2, actions_2 = to_df(comb_data_2, set2_names)
    dfs = [[body_and_box_df_1, body_and_box_df_2, 'handbox']]
    
    # PCA
    #print('Using PCA')
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(body_and_box_df_1.T)
    pca = PCA(n_components = 0.95)
    pca.fit(data_rescaled)
    reduced = pca.transform(data_rescaled)
    
    if use_PCA:
        predictors = pd.DataFrame(reduced, index=actions_1)
    else:
        predictors = body_and_box_df_1.T
    
    # Construct z-scored dissimilarity matrices for each predictor
    #print('Construcing dissimilarity matrices')
    predictors_matrix = np.zeros((60,60))
    for i in range(predictors.shape[1]): # cycle through each predictor
        values = [[val] for val in predictors[predictors.columns[i]].tolist()]
        distance_vector = pdist(values, metric='sqeuclidean')
        distance_vector = zscore(distance_vector)
        distance_matrix = squareform(distance_vector)
        predictors_matrix = predictors_matrix + distance_matrix
    
    # Cross-Validation
    
    alpha = 0.01
    print('Cross-validating using elastic regression at alpha = {}'.format(alpha))

    for behavior in behavioral_data:
        
        if reg_regression:
            net = ElasticNet(alpha=alpha)
        else:
            net = LinearRegression()
        
        judgements = np.ravel(zscore(behavioral_data[behavior]))
        judgements_matrix = squareform(judgements)
        
        predicted = []
        scores = []
        
        # Regularized Regression
        for i in range(len(predictors)):
            behavior_train, target = hold_pairs(i, judgements_matrix)
            feature_train, feature_test = hold_pairs(i, predictors_matrix)
            net.fit(feature_train, behavior_train)
            predicted = net.predict(feature_test)
            scores.append(pearsonr(np.ravel(predicted), np.ravel(target))[0])
        
        #predicted = np.ravel(np.array(predicted)); target = np.ravel(np.array(target))
        prediction_accuracy = np.mean(scores)
        #set_trace()
        print('Average prediction accuracy for {} similarity: {}'.format(behavior, prediction_accuracy))
    
    print('----------')
    
    # Analysis requiring unraveled frame data
    # Plots
    '''
    dir = '{}_plots'.format(label)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    for df in dfs: # df = ['handbox' or 'handpose', df for set 1, df for set 2]
        
        cl = correlate(df[0].corr(), df[1].corr()) # Calculate correlation matrix between video sets 1 and 2's dissimilarity matrices
        # n = df[0].corrwith(df[1], axis=0)
        
        #print('Kendall\'s tau for body pose and {} features between set 1 and 2 = {}'.format(df[2], kendalltau(arr_set_1, arr_set_2)[0]))
        print('Correlation for body pose and {} features between set 1 and 2 = {}'.format(df[2], cl.mean()))
        
        # Plots
        for m in methods:
            for i in range(2):
                order = plot_heir_cluster(df[i].T, m, dir, i+1)
                hierarchy = [list(df[i].columns)[j] for j in order]
                sorted = df[i].reindex(columns=hierarchy) # Arrange DF according to hierarchical clustering solution
                plot_heatmap(sorted, m, dir, i+1) # Plot correlation heatmap of sorted DF
    '''
