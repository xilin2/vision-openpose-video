import sys
import numpy as np
import scipy.io
import pandas as pd
import os

import analysis_util as util

from sklearn.datasets import load_iris
from scipy.signal import correlate
from scipy.stats import zscore, kendalltau, pearsonr, wilcoxon
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet

from pdb import set_trace

class BehavioralPoseDataAnalysis():

    '''
    Class which loads, simplifies, and performs analyses on pose and
    behavioral data
    '''

    def __init__(self, body_file, hand_file, xls_path, model, set_num, behaviors=None):
    
        body, hand, set1_names, set2_names = self.load_pose_data(body_file, hand_file, xls_file)
        if behaviors:
            self.behavioral_data = self.load_behavioral_data(behaviors)
        self.model = model
        self.data_dict, data = self.simplify_data(body, hand, set1_names, set2_names, set_num)
        self.data = data[0]; self.names = data[1]
    
    
    def load_pose_data(self, body_file, hand_file, xls_file):
        
        '''
        Loads pose and video data. Returns body, hand, and video name data
        as dictionaries.
        '''
        
        body = scipy.io.loadmat(body_file)
        hand = scipy.io.loadmat(hand_file)
        xls = pd.ExcelFile(xls_file)
        
        # Delete extra dictionary keys
        del body['__header__']; del hand['__header__']
        del body['__version__']; del hand['__version__']
        del body['__globals__']; del hand['__globals__']
        
        set1_names =  xls.parse(xls.sheet_names[0])
        set2_names =  xls.parse(xls.sheet_names[1])
        
        return body, hand, set1_names, set2_names
    
    def load_behavioral_data(self, behaviors):
        
        '''
        Loads requested behavioral data. Stores data in dictionary with
        keys as the behavioral category.
        '''
        
        behavioral_data = {}
        for b in behaviors:
            if b == 'goals':
                behavioral_data['Goals'] =  scipy.io.loadmat('data/GoalSimilarity.mat')['GuidedBehaviorModel'][0][0][1]
            elif b == 'intuitive':
                behavioral_data['Intuitive'] = scipy.io.loadmat('data/IntuitiveActionSim.mat')['IntuitiveSimilarity'][0][0][1][0][0][1]
            elif b == 'movement':
                behavioral_data['Movement'] = scipy.io.loadmat('data/MovementSimilarity.mat')['GuidedBehaviorModel'][0][0][1]
            elif b == 'visual':
                behavioral_data['Visual'] =  scipy.io.loadmat('data/VisualSimilarity.mat')['GuidedBehaviorModel'][0][0][1]

        return behavioral_data
    
    def simplify_data(self, body, hand, set1_names, set2_names, set_num):
    
        '''
        Modifies data according to the data's purpose. The three options
        are to average the data, remove extraneous information (such as
        score, ID) and convert hand box data (x,y,w,h) to 4 coordinates,
        and to simply unravel all data. Returns a combined hand and body
        data set.
        '''
    
        orgd_body = {}; orgd_hand = {}
    
        # Averages coordinates across 75 frames. Each video will have 26 values (18 for body, 8 for hand)
        if self.model == 'avg':
            for key in body:
                orgd_body[key] = util.average_body(util.clean_body(body[key]))
                orgd_hand[key] = util.average_hands(util.clean_hands(hand[key]))

        # Used for Procrustes, converts data set to coordinates-only data.
        # Saves hands video data set, organized by frame and with hand box
        # data (x,y,w,h) converted to four coordinates. Saves body video
        # data set, organized by frame and with score and ID removed).
        # Each of 75 frames will have 26 values.
        elif self.model == 'pro':
            for key in body:
                orgd_body[key] = np.ravel(util.clean_body(body[key])).tolist()
                orgd_hand[key] = np.ravel(util.clean_hands(hand[key])).tolist()
              
        # Unravel data under each dictionary key.
        else:
            for key in body:
                orgd_body[key] = []; orgd_hand[key] = []
                for frame_index in range(len(body[key])):
                    for el in body[key][frame_index]:
                        orgd_body[key].append(el)
                    for el in hand[key][frame_index]:
                        orgd_hand[key].append(el)

        # Combine body and hand dictionaries
        comb_data_1, comb_data_2 = util.combine_body_hand(orgd_body, orgd_hand)
        
        if set_num == 1:
            return comb_data_1, util.to_df(comb_data_1, set1_names)
        else:
            return comb_data_2, util.to_df(comb_data_2, set2_names)
    
    def get_split_half_reliabilities(self):
    
        '''
        Estimate and prints split-half-reliability of dataset. Repeats
        reliability test 1000 times and averages results.
        '''
    
        print('Printing reliabilities... may take a while')
        reliabilities = util.get_split_half_reliabilities(1000, self.data)
        #    for el in sorted(reliabilities.items(), key=lambda x:x[1]):
        #        print(el)
        values_only = [reliabilities[i] for i in reliabilities]
        print('Reliability mean: {}'.format(np.mean(values_only)))
        print('Reliability median: {}'.format(np.median(values_only)))
    
    def cross_validation(self, method=None, use_PCA=False, abso=False):
            
        '''
        Uses either regularized or linear regression to cross-validate
        pose data and behavioral data. Pose data is organized into
        predictors  of dissimilarity matrices of size (60, 60). Prints the
        mean of the prediction accuracies, and returns a dictionary of the
        arrays of prediction accuracies organized by behavioral category.
        '''
        
        predictors = self.get_predictors(use_PCA)
        
        # Creates array of dissimilarity matrices of pose data to be used as train/test data
        if self.model == 'pro':
            predictors_matrix = self.construct_procrustes_matrices(predictors)
        else:
            predictors_matrix = self.construct_distance_matrices(predictors)
            
        print('Cross-validation results for {}'.format(self.model))

        scores_for_behavior = {} # Stores scores array for each behavioral
            # data category analyzed
        
        for behavior in self.behavioral_data:
            
            # Initializes regression net
            if method == 'reg':
                alpha = 0.05
                net = ElasticNet(alpha=alpha)
            else:
                net = LinearRegression()
            
            # Creates array of dissimilarity matrices of behavioral data
            # to be used as train/target data
            judgements = np.ravel(zscore(self.behavioral_data[behavior]))
            judgements_matrix = squareform(judgements)
            
            #abs_scores = []
            scores = []
            
            # Cycles through each video, obtaining prediction accuracies
            for i in range(60):
                behavior_train, target = util.hold_pairs(i, judgements_matrix)
                feature_train = []; feature_test = []
                # Appends train/test data from each predictor onto arrays
                for j, pred in enumerate(predictors_matrix):
                    tr, te = util.hold_pairs(j, pred)
                    feature_train.append(tr); feature_test.append(te)
                net.fit(np.transpose(feature_train), np.transpose(behavior_train))
                predicted = net.predict(np.transpose(feature_test))
                score = pearsonr(np.ravel(predicted), np.ravel(target))[0]
                if abso:
                    scores.append(abs(score))
                else:
                    scores.append(score)
            
            scores_for_behavior[behavior] = scores
            prediction_accuracy = np.mean(scores)
            #abs_prediction_accuracy = np.mean(abs_scores)
            
            print('Average prediction accuracy for {} similarity: {}'.format(behavior, prediction_accuracy))
            
        print('----------')
        
        return scores_for_behavior
    
    def get_predictors(self, use_PCA):
    
        '''
        Returns either a DataFrame of PCA 0.95 data or a DataFrame of the
        original data. Unless the model used is Procrustes, these
        predictors will be used to construct the dissimilarity matrices.
        '''
    
        if use_PCA:
            scaler = MinMaxScaler()
            data_rescaled = scaler.fit_transform(self.data.T)
            pca = PCA(n_components = 0.95)
            pca.fit(data_rescaled)
            reduced = pca.transform(data_rescaled)
            predictors = pd.DataFrame(reduced, index=actions_1)
        else:
            predictors = self.data.T
        
        return predictors
    
    def construct_distance_matrices(self, predictors):
    
        '''
        Constructs dissimilarity matrice using the squared euclidean
        distance between values for each predictor in each video. Return
        an array of dissimilarity matrices of len = number of predictors.
        '''
    
        predictors_matrix = [] # stores dissimilarity matrices
        
        for i in range(0, predictors.shape[1], 2): # cycle through each predictor
            #values = [[val] for val in  predictors[predictors.columns[i]].tolist()]
            values = [val for val in  predictors[[i,i+1]].values.tolist()]
            distance_vector = pdist(values, metric='sqeuclidean')
            distance_vector = zscore(distance_vector)
            distance_matrix = squareform(distance_vector)
            predictors_matrix.append(distance_matrix)
        
        return predictors_matrix
    
    def construct_procrustes_matrices(self, predictors):
    
        '''
        Constructs dissimilarity matrice by finding the distance between
        the trajectory of a point in one video, and the Procrustes
        transformation of the same point in another video. Returns an
        array of dissimilarity matrices of len = 26 (18 body
        points/trajectories, 8 hand box points)
        '''
    
        print('Calculating Procrustes distances... may take a while (~1 min)')
        
        n = 26 # number of trajectories / points describing each video
        trajectories = {}
        
        # Get n=26 trajectory sets for all videos
        for video in self.data_dict:
            trajectories[video] = util.get_trajectories(self.data_dict[video])
        
        # Construct dissimilarity matrix based on average distance between
        # two videos' 26 Procrustes transformed trajectories.
        predictors_matrix = [] # stores dissimilarity matrices
        for k in range(n): # Cycles through each of 26 trajectories
            # distance matrix to later be added to predictors_matrix
            distance_for_feature = [[] for m in range(60)]
            for i, vid1 in enumerate(trajectories):
                for j, vid2 in enumerate(trajectories):
                    avg_dist = []
                    if i == j: # same video
                        distance_for_feature[i].append(0)
                        continue
            
                    traj1 = trajectories[vid1][k]; traj2 = trajectories[vid2][k]
                    
                    # In the case that either trajectory does not contain
                    # >1 unique points, modify first point by 0.01
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
            
        return predictors_matrix
    
    def wilcoxon_test(self, scores):
    
        '''
        Performs and prints results of Wilcoxon test on prediction
        accuracy scores.
        '''
    
        print('Wilcoxon results for {} model'.format(self.model))
        for behavior in scores:
            print('{} similarity: {}'.format(behavior, wilcoxon(scores[behavior], alternative='greater')))
#        print('Two-Sided Movement and Visual Similarity: {}'.format(wilcoxon(scores['Visual'], scores['Movement'], alternative='two-sided')))
#        print('Two-Sided Goals and Movement Similarity: {}'.format(wilcoxon(scores['Movement'], scores['Goals'], alternative='two-sided')))
            
        print('----------')

if __name__ == '__main__':
    
    body_file = 'data/vids_body_new_track.mat'
    hand_file = 'data/vids_hand_new_track.mat'
    xls_file = 'data/vidnamekey.xlsx'
    
    complete_scores = {}
    
    for model in ['avg', 'pro']:
        analysis = BehavioralPoseDataAnalysis(body_file, hand_file, xls_file, model, 1, ['intuitive'])
        scores = analysis.cross_validation(method='reg', abso=True)
        analysis.wilcoxon_test(scores)
        complete_scores[model] = scores
    
    util.plot_error_bar(complete_scores)
    
    # Analysis requiring unraveled frame data
    # Plots
    '''
    
    label = 'empty'
    methods = ['complete', 'average', 'weighted', 'ward']
    
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
