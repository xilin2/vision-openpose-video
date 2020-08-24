import sys
import numpy as np
import scipy.io
import pandas as pd
import os

import analysis_util as util

from sklearn.datasets import load_iris
from scipy.signal import correlate
from scipy.stats import zscore, kendalltau, pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet

from pdb import set_trace

average_data = False
original_data = False
procrustes_dist = False
use_PCA = False
reg_regression = False

args = sys.argv[1:]

for arg in args:
    if arg == 'average':
        average_data = True
    elif arg == 'original':
        original_data = True
    elif arg.lower() == 'pca':
        use_PCA = True
    elif arg == 'regularized':
        reg_regression = True
    elif arg == 'procrustes':
        procrustes_dist = True
    else:
        print('\'{}\' argument not recognized. Valid arguments are \'average\', \'original\', \'pca\', \'procrustes\', and \'regularized\''.format(arg))
        sys.exit()
        
#label = 'empty'
#methods = ['complete', 'average', 'weighted', 'ward']

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
    for key in body:
        orgd_body[key] = util.average_body(util.clean_body(body[key]))
        orgd_hand[key] = util.average_hands(util.clean_hands(hand[key]))

# Clean data to remove ids, scores, and w/h measurements
else:
    for key in body:
        orgd_body[key] = np.ravel(util.clean_body(body[key])).tolist()
        orgd_hand[key] = np.ravel(util.clean_hands(hand[key])).tolist()

# Combine body and hand dictionaries
comb_data_1, comb_data_2 = util.combine_body_hand(orgd_body, orgd_hand)

# Reliabilities
'''
print('Getting reliabilities... may take a while')
reliabilities = util.get_split_half_reliabilities(1000, comb_data_1)
#    for el in sorted(reliabilities.items(), key=lambda x:x[1]):
#        print(el)
values_only = [reliabilities[i] for i in reliabilities]
print('Reliability mean: {}'.format(np.mean(values_only)))
print('Reliability median: {}'.format(np.median(values_only)))
'''

# Create DataFrames corresponding to different video sets and hand data sets
#print('Creating dataframes')
body_and_box_df_1, actions_1 = util.to_df(comb_data_1, set1_names)
body_and_box_df_2, actions_2 = util.to_df(comb_data_2, set2_names)
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

# Get trajectory sets for all videos
if procrustes_dist:
    
    print('Calculating Procrustes distances... may take a while (~1 min)')
    
    n = 26 # number of trajectories / points describing each video
    trajectories = {}
        
    for video in comb_data_1:
        trajectories[video] = util.get_trajectories(comb_data_1[video])
    
    # Construct dissimilarity matrix based on average distance between two videos' 26 Procrustes transformed trajectories
    predictors_matrix = []
    for k in range(n):
        distance_for_feature = [[] for m in range(60)]
        for i, vid1 in enumerate(trajectories):
            for j, vid2 in enumerate(trajectories):
                avg_dist = []
                if i == j:
                    distance_for_feature[i].append(0)
                    continue
                traj1 = trajectories[vid1][k]; traj2 = trajectories[vid2][k]
                # In the case that either trajectory does not contain >1 unique points, modify first point by 0.01
                if traj1.count(traj1[0]) == len(traj1):
                    traj1[0] = [traj1[0][0]-0.01, traj1[0][1]-0.01]
                if traj2.count(traj2[0]) == len(traj2):
                    traj2[0] = [traj2[0][0]-0.01, traj2[0][1]-0.01]
                mtx1_1, mtx2_1, disp = procrustes(traj1, traj2)
                mtx1_2, mtx2_2, disp = procrustes(traj2, traj1)
                avg_dist.append(util.find_procrustes_distance(mtx1_1, mtx2_1))
                avg_dist.append(util.find_procrustes_distance(mtx1_2, mtx2_2))
                distance_for_feature[i].append(np.mean(avg_dist))
        
        distance_for_feature = zscore(squareform(distance_for_feature))
        distance_for_feature = squareform(distance_for_feature)
        predictors_matrix.append(distance_for_feature)
    
    #set_trace()
        

# Construct z-scored dissimilarity matrices for each predictor
else:
    predictors_matrix = []
    # predictors_matrix = np.zeros((60,60))
    for i in range(predictors.shape[1]): # cycle through each predictor
        values = [[val] for val in predictors[predictors.columns[i]].tolist()]
        distance_vector = pdist(values, metric='sqeuclidean')
        distance_vector = zscore(distance_vector)
        distance_matrix = squareform(distance_vector)
        predictors_matrix.append(distance_matrix)

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
    
    for i in range(60):
        behavior_train, target = util.hold_pairs(i, judgements_matrix)
        feature_train = []; feature_test = []
        for j, pred in enumerate(predictors_matrix):
            tr, te = util.hold_pairs(j, pred)
            feature_train.append(tr); feature_test.append(te)
        #set_trace()
        net.fit(np.transpose(feature_train), np.transpose(behavior_train))
        predicted = net.predict(np.transpose(feature_test))
        scores.append(pearsonr(np.ravel(predicted), np.ravel(target))[0])
        
#    for i in range(len(predictors)):
#        behavior_train, target = util.hold_pairs(i, judgements_matrix)
#        feature_train, feature_test = util.hold_pairs(i, predictors_matrix)
#        net.fit(feature_train, behavior_train)
#        predicted = net.predict(feature_test)
#        scores.append(kendalltau(np.ravel(predicted), np.ravel(target))[0])
    
    #set_trace()
    
    prediction_accuracy = np.mean(scores)
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
            order = util.plot_heir_cluster(df[i].T, m, dir, i+1)
            hierarchy = [list(df[i].columns)[j] for j in order]
            sorted = df[i].reindex(columns=hierarchy) # Arrange DF according to hierarchical clustering solution
            util.plot_heatmap(sorted, m, dir, i+1) # Plot correlation heatmap of sorted DF
'''
