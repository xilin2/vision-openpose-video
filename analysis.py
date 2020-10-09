import sys
import numpy as np
import scipy.io
import pandas as pd
import os
import matplotlib.pyplot as plt

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

import statsmodels.formula.api as smf
import statsmodels.api as sm
#import statsmodels.discrete.discrete_model

from pdb import set_trace

def cross_validation(predictors, judgements, label, removed_joints=[], removed_videos=[], method=None):
           
    '''
    Uses either regularized or linear regression to cross-validate
    pose data and behavioral data. Pose data is organized into
    predictors  of dissimilarity matrices of size (60, 60). Prints the
    mean of the prediction accuracies, and returns a dictionary of the
    arrays of prediction accuracies organized by behavioral category.
    '''
   
    #print('Cross-validation results for {}'.format(label))
    scores_for_behavior = {} # Stores scores array for each behavioral
       # data category analyzed
       
    # Initializes regression net
    if method == 'reg':
        alpha = 0.05
        net = ElasticNet(alpha=alpha)
    else:
        net = LinearRegression()
   
    # Creates array of dissimilarity matrices of behavioral data
    # to be used as train/target data
    judgements = np.ravel(zscore(self.group_behavioral_data[behavior]))
    judgements_matrix = squareform(judgements)

    scores = []
   
    # Cycles through each video, obtaining prediction accuracies
    for i in range(60):
        if i in removed_videos:
            continue
        behavior_train, target = util.hold_pairs(i, judgements, removed_videos)
        feature_train = []; feature_test = []
        # Appends train/test data from each predictor onto arrays
        for j, pred in enumerate(predictors_matrix):
            if j in removed_joints:
                continue
            tr, te = util.hold_pairs(i, pred, removed_videos)
            feature_train.append(tr); feature_test.append(te)
        net.fit(np.transpose(feature_train), np.transpose(behavior_train))
        predicted = net.predict(np.transpose(feature_test))
        score = pearsonr(np.ravel(predicted), np.ravel(target))[0]
        scores.append(score)
   
    scores_for_behavior[behavior] = scores
    prediction_accuracy = np.mean(scores)
   
    print('Average prediction accuracy for {} similarity: {}'.format(label, prediction_accuracy))
       
    print('----------')
           
    return scores_for_behavior

class Behaviors():

    def __init__(self, behaviors):
    
        self.group_behavioral_data = {}
        self.single_behavioral_data = {}
        
        for b in behaviors:
            if b == 'goals':
                dat = scipy.io.loadmat('data/GoalSimilarity.mat')['GuidedBehaviorModel'][0][0]
                self.group_behavioral_data['Goals'] = dat[1]
                self.single_behavioral_data['Goals'] = dat[2]
            elif b == 'intuitive':
                self.group_behavioral_data['Intuitive'] = scipy.io.loadmat('data/IntuitiveActionSim.mat')['IntuitiveSimilarity'][0][0][1][0][0][1]
            elif b == 'movement':
                dat = scipy.io.loadmat('data/MovementSimilarity.mat')['GuidedBehaviorModel'][0][0]
                self.group_behavioral_data['Movement'] = dat[1]
                self. single_behavioral_data['Movement'] = dat[2]
            elif b == 'visual':
                dat = scipy.io.loadmat('data/VisualSimilarity.mat')['GuidedBehaviorModel'][0][0]
                self.group_behavioral_data['Visual'] = dat[1]
                self.single_behavioral_data['Visual'] = dat[2]
        
class LayerDataAnalysis():

    def __init__(self, layer_data_file, behaviors=None):
        self.layer_data = self.load_layer_data(layer_data_file)
        if behaviors:
            beh = Behaviors(behaviors)
            self.group_behavioral_data = beh.group_behavioral_data
            self.single_behavioral_data = beh.single_behavioral_data
    
    def load_layer_data(self, file):
    
        '''
        Loads pose and video data. Returns body, hand, and video name data
        as dictionaries.
        '''
        
        layers = scipy.io.loadmat(file)
        numerical_layers = {}
        
        # Delete extra dictionary keys
        del layers['__header__']
        del layers['__version__']
        del layers['__globals__']
        
        for layer in layers:
            l = layer.split('_')[1]
            numerical_layers[l] = layers[layer][0]
        
        return numerical_layers
    
    def cross_validation(self, method=None):
        
        scores_for_behavior = {} # Stores scores array for each behavioral
            # data category analyzed
        
        for behavior in self.group_behavioral_data:
            
            
            # Initializes regression net
            if method == 'reg':
                alpha = 0.05
                net = ElasticNet(alpha=alpha)
            else:
                net = LinearRegression()
            
            # Creates array of dissimilarity matrices of behavioral data
            # to be used as train/target data
            judgements = np.ravel(zscore(self.group_behavioral_data[behavior]))
            judgements_matrix = squareform(judgements)

            scores_by_layer = {}
            
            # Cycles through each video, obtaining prediction accuracies
            for layer in self.layer_data:
#                if layer == '55':
#                    continue
                scores = [] # temporarily holds score data for each video
                for i in range(60):
                    behavior_train, target = util.hold_pairs(i, judgements_matrix)
                    feature_train, feature_test = util.hold_pairs(i, squareform(self.layer_data[layer]))
                    feature_train = np.array(feature_train).reshape(1,-1)
                    feature_test = np.array(feature_test).reshape(1,-1)
                    net.fit(np.transpose(feature_train), np.transpose(behavior_train))
                    predicted = net.predict(np.transpose(feature_test))
                    score = pearsonr(np.ravel(predicted), np.ravel(target))[0]
                    scores.append(score)
                scores_by_layer[layer] = scores
                
            scores_for_behavior[behavior] = scores_by_layer
            #prediction_accuracy = np.mean(scores)
            #abs_prediction_accuracy = np.mean(abs_scores)
            
            #print('Average prediction accuracy for {} similarity: {}'.format(behavior, prediction_accuracy))
            
        print('----------')
        
        return scores_for_behavior
    
    def regression_analysis(self, scores, deg, behavior):
    
        means = np.array([np.nanmean(scores[layer]) for layer in scores])
        df = pd.DataFrame(means, columns=['means'])
        df['layers'] = df.index
#        k = np.arange(len(means))
        reg = {'Means': means}
#
#        for d in deg:
#            fit = np.polyfit(np.ravel(np.argwhere(~np.isnan(means))), means[~np.isnan(means)], d)
#            fit_fn = np.poly1d(fit)
#            reg[d] = fit_fn(k)
        
#        fit = np.polyfit(np.log(np.ravel(np.argwhere(~np.isnan(means)))), means[~np.isnan(means)], 1)
#        fit_fn = np.poly1d(fit)
#        reg['log'] = fit_fn(k)
        
        print('Linear OLS regression for {}'.format(behavior))
        linear = smf.ols(formula='means ~ layers', data=df, missing='drop').fit()
        reg['Linear'] = linear.predict()
        print(linear.summary())
        
        print('Quadratic OLS regression for {}'.format(behavior))
        quad = smf.ols(formula='means ~ layers + layers^2', data=df, missing='drop').fit()
        reg['Quadratic'] = quad.predict()
        print(quad.summary())
        
        log = smf.Logit(formula='means ~ layers', data=df, missing='drop').fit()
        reg['Log'] = log.predict()
        print('Anova table for {}'.format(behavior))
        ano = sm.stats.anova_lm(linear, quad)
        print(ano)
        
        
#        df = pd.DataFrame(columns = ['value', 'layer', 'regression_type'])
#        for reg_type in reg:
#            for value in reg[reg_type]:
#                df = df.append({'value': value, 'regression_type': reg_type}, ignore_index=True)
#
#        model = smf.ols(formula='value ~ regression_type', data=df, missing='drop').fit()
#        print(model.summary())
        
        print('----------')
        
        return reg
        

class PoseDataAnalysis():

    '''
    Class which loads, simplifies, and performs analyses on pose and
    behavioral data
    '''

    def __init__(self, body_file, hand_file, xls_path, removed_videos=[], removed_joints=[], model=None, set_num=1, behaviors=None):
    
        body, hand, set1_names, set2_names = self.load_pose_data(body_file, hand_file, xls_file)
        if behaviors:
            beh = Behaviors(behaviors)
            self.group_behavioral_data = beh.group_behavioral_data
            self.single_behavioral_data = beh.single_behavioral_data
        self.model = model
        self.removed_videos = removed_videos
        self.removed_joints = removed_joints
        
        # self.fbf = dict of size 60 holding an array of (75x52)
        # self.data_dict = self.data as a dict with numerical video names
        # self.data = dataframe with action names
        self.fbf, self.data_dict, data = self.simplify_data(body, hand, set1_names, set2_names, set_num)
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
        
        group_behavioral_data = {}
        single_behavioral_data = {}
        
        for b in behaviors:
            if b == 'goals':
                dat = scipy.io.loadmat('data/GoalSimilarity.mat')['GuidedBehaviorModel'][0][0]
                group_behavioral_data['Goals'] = dat[1]
                single_behavioral_data['Goals'] = dat[2]
            elif b == 'intuitive':
                group_behavioral_data['Intuitive'] = scipy.io.loadmat('data/IntuitiveActionSim.mat')['IntuitiveSimilarity'][0][0][1][0][0][1]
            elif b == 'movement':
                dat = scipy.io.loadmat('data/MovementSimilarity.mat')['GuidedBehaviorModel'][0][0]
                group_behavioral_data['Movement'] = dat[1]
                single_behavioral_data['Movement'] = dat[2]
            elif b == 'visual':
                dat = scipy.io.loadmat('data/VisualSimilarity.mat')['GuidedBehaviorModel'][0][0]
                group_behavioral_data['Visual'] = dat[1]
                single_behavioral_data['Visual'] = dat[2]

        return group_behavioral_data, single_behavioral_data
    
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
                orgd_body[key] = util.clean_body(body[key])
                orgd_hand[key] = util.clean_hands(hand[key])
                
        else:
            orgd_body = body; orgd_hand = hand

        # Get frame by frame full data
        frame_by_frame_1, frame_by_frame_2 = util.get_frame_by_frame(body, hand)

        # Combine body and hand dictionaries
        comb_data_1, comb_data_2 = util.combine_body_hand(orgd_body, orgd_hand)
        
        if set_num == 1:
            return frame_by_frame_1, comb_data_1, util.to_df(comb_data_1, set1_names)
        else:
            return frame_by_frame_2, comb_data_2, util.to_df(comb_data_2, set2_names)
    
    def get_split_half_reliabilities(self):
    
        '''
        Estimate and prints split-half-reliability of dataset. Repeats
        reliability test 1000 times and averages results.
        '''
    
        print('Printing reliabilities... may take a while')
        reliabilities = {}
        for group in self.fbf:
            #reliabilities = util.get_split_half_reliabilities(1000, self.data)
            reliabilities[group] = util.get_split_half_reliabilities(1000, self.fbf[group])
        #    for el in sorted(reliabilities.items(), key=lambda x:x[1]):
        #        print(el)
        values_only = [reliabilities[i] for i in reliabilities]
        print('Reliability mean: {}'.format(np.mean(values_only)))
        print('Reliability median: {}'.format(np.median(values_only)))
    
    def cross_validation(self, predictors, method=None, use_PCA=False, abso=False):
            
        '''
        Uses either regularized or linear regression to cross-validate
        pose data and behavioral data. Pose data is organized into
        predictors  of dissimilarity matrices of size (60, 60). Prints the
        mean of the prediction accuracies, and returns a dictionary of the
        arrays of prediction accuracies organized by behavioral category.
        '''
        
        predictors = self.get_predictors(use_PCA)
        
        # Creates array of dissimilarity matrices of pose data to be used as train/test data
        
        print('Pose cross-validation results for {}'.format(self.model))

        scores_for_behavior = {} # Stores scores array for each behavioral
            # data category analyzed
        
        for behavior in self.group_behavioral_data:
            
            # Initializes regression net
            if method == 'reg':
                alpha = 0.05
                net = ElasticNet(alpha=alpha)
            else:
                net = LinearRegression()
            
            # Creates array of dissimilarity matrices of behavioral data
            # to be used as train/target data
            judgements = np.ravel(zscore(self.group_behavioral_data[behavior]))
            judgements_matrix = squareform(judgements)
            
            #abs_scores = []
            scores = []
            
            # Cycles through each video, obtaining prediction accuracies
            for i in range(60):
                if i in self.removed_videos:
                    continue
                behavior_train, target = util.hold_pairs(i, judgements_matrix, self.removed_videos)
                feature_train = []; feature_test = []
                # Appends train/test data from each predictor onto arrays
                for j, pred in enumerate(predictors_matrix):
                    if j in self.removed_joints:
                        continue
                    tr, te = util.hold_pairs(i, pred, self.removed_videos)
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
        
        if self.model == 'pro':
            predictors_matrix = self.construct_procrustes_matrices(predictors)
            self.check_procrustes_validity(predictors_matrix)
        else:
            predictors_matrix = self.construct_distance_matrices(predictors)
        
        return predictors_matrix
    
    def construct_distance_matrices(self, predictors):
    
        '''
        Constructs dissimilarity matrice using the squared euclidean
        distance between values for each predictor in each video. Return
        an array of dissimilarity matrices of len = number of predictors.
        '''
    
        predictors_matrix = [] # stores dissimilarity matrices
        
        for i in range(0, predictors.shape[1], 2): # cycle through each predictor
            #values = [[val] for val in  predictors[predictors.columns[i]].tolist()]
            values = [val for val in predictors[[i,i+1]].values.tolist()]
            distance_vector = pdist(values, metric='sqeuclidean')
            distance_vector = zscore(distance_vector, nan_policy='omit')
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
        for i, video in enumerate(self.data):
            trajectories[video] = util.get_trajectories(list(self.data[video]))
        
        # Construct dissimilarity matrix based on average distance between
        # two videos' 26 Procrustes transformed trajectories.
        predictors_matrix = [] # stores dissimilarity matrices
        for k in range(n): # Cycles through each of 26 trajectories
        
            # distance matrix to later be added to predictors_matrix
            distance_for_feature = [[] for m in range(len(trajectories))]
            for i, vid1 in enumerate(trajectories):
            
                for j, vid2 in enumerate(trajectories):
                    avg_dist = []
                    
                    if i == j: # same video
                        distance_for_feature[i].append(0)
                        continue
            
                    traj1 = trajectories[vid1][k]; traj2 = trajectories[vid2][k]
                    
                    # Check for empty trajectoriess
                    # nanchange
#                    if traj1.count([0,0]) == len(traj1):
#                        distance_for_feature[i].append(np.nan)
#                        continue
#                    if traj2.count([0,0]) == len(traj2):
#                        distance_for_feature[i].append(np.nan)
#                        continue
                    
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
            
            distance_for_feature = zscore(squareform(distance_for_feature, checks=False), nan_policy='omit')
            distance_for_feature = squareform(distance_for_feature)
            
#            plt.imshow(distance_for_feature, cmap='hot', interpolation='nearest')
#            plt.xticks(np.arange(len(trajectories)), trajectories.keys(), size=7, rotation=90)
#            plt.yticks(np.arange(len(trajectories)), trajectories.keys(), size=7)
#            plt.title(k)
#            plt.gcf().set_size_inches((10,10))
#            plt.subplots_adjust(left=0.22, bottom=0.22)
#            plt.savefig('plots/joint_{}_visualization.png'.format(k))
#            plt.close()
            
            predictors_matrix.append(distance_for_feature)
            
        return predictors_matrix
    
    def check_procrustes_validity(self, predictors):
        # add all matrices together
        np_predictors = [np.array(p) for p in predictors]
        sum_predictor = np_predictors[0]
        for i in range(1, len(np_predictors)):
            #sum_predictor += np_predictors[i]
            sum_predictor = util.sum_two_matrices(sum_predictor, np_predictors[i])
        
        values = []
        hands_only = [7,8,11,13,17,18,19,20,21,22,31,34,38,39,41,56,58]
        full_body = [0,1,2,3,4,5,6,7,10,16,24,25,26,27,28,29,36,37,42,43,44,45,46,47,48,51,53,54,55,59]
        no_hands = []
        only_body = []
        
        for i in range(len(sum_predictor)):
            if i in self.removed_videos:
                continue
            for j in range(i, len(sum_predictor[0])):
                if j in self.removed_videos:
                    continue
                if i not in hands_only and j not in hands_only:
                    no_hands.append([(i+1,j+1), sum_predictor[i][j]])
                if i in full_body and j in full_body:
                    only_body.append([(i+1,j+1), sum_predictor[i][j]])
                values.append([(i+1,j+1), sum_predictor[i][j]])
        
        values.sort(key=lambda x: x[1])
        no_hands.sort(key=lambda x: x[1])
        only_body.sort(key=lambda x: x[1])
        
        set_trace()
    
    def count_joint_frequency(self):
        
        df = pd.DataFrame(columns = ['video', 'joint', 'frequency'])
        for joint in range(26):
            for i, video in enumerate(self.data_dict):
                #for frame in self.fbf[video]:
                freq = 0
                for j in range(52):
                    frame = self.data_dict[video][j*52:j*52+52]
                    if int(frame[joint*2]) != 0 or int(frame[joint*2+1]) != 0:
                        freq += 1
                df = df.append({'video': self.names[i], 'joint': joint, 'frequency': freq}, ignore_index = True)
                
        videos = df['video'].unique()
        joints = df['joint'].unique()
        
        vid_values = {v: 0 for v in videos}
        joint_values = {j: 0 for j in joints}
        
        for i, row in df.iterrows():
            if row['frequency'] > 0:
                joint_values[row['joint']] += 1
                
        removed_joints = []
#        for i in range(len(joint_values)):
#            if joint_values[i] < 30:
#                removed_joints.append(i)
        
        for i, row in df.iterrows():
            if row['frequency'] > 0:
#                if row['joint'] not in removed_joints:
#                    vid_values[row['video']] += 1
                vid_values[row['video']] += 1
        
        removed_videos = []
        for i, video in enumerate(self.data):
            #if vid_values[video] < (len(joint_values)-len(removed_joints)):
            if vid_values[video] < 5:
                removed_videos.append(i)

        return df, removed_videos, removed_joints
    
    def wilcoxon_test(self, scores):
    
        '''
        Performs and prints results of Wilcoxon test on prediction
        accuracy scores.
        '''
    
        print('Wilcoxon results for {} model'.format(self.model))
        for behavior in scores:
            print('{} similarity: {}'.format(behavior, wilcoxon(scores[behavior], alternative='greater')))
        if 'Movement' in scores and 'Visual' in scores:
            print('Two-Sided Movement and Visual Similarity: {}'.format(wilcoxon(scores['Visual'], scores['Movement'], alternative='two-sided')))
        if 'Movement' in scores and 'Goals' in scores:
            print('Two-Sided Goals and Movement Similarity: {}'.format(wilcoxon(scores['Movement'], scores['Goals'], alternative='two-sided')))
            
        print('----------')
    
    def ols_reg_comparison(self, scores):
        
        '''
        Performs and prints an OLS regression between scores, and
        model_type, analysis_type, and the interaction term between these
        two.
        '''
        
        print('OLS regression')
        
        df = pd.DataFrame(columns = ['score', 'model_type', 'analysis_type'])
        
        model_types = {'avg': 0, 'pro': 1}
        analysis_types = {'Visual': 0, 'Movement': 1, 'Goals': 2}
        
        # Create dataframe
        for model_type in scores:
            for analysis_type in scores[model_type]:
                for score in scores[model_type][analysis_type]:
                    df = df.append({'score': score, 'model_type': model_types[model_type], 'analysis_type': analysis_types[analysis_type]}, ignore_index=True)
        
        model = smf.ols(formula='score ~ model_type + analysis_type + model_type:analysis_type', data=df).fit()
        print(model.summary())
        
        print('----------')
        

if __name__ == '__main__':
    
    body_file = 'data/vids_body_new_track.mat'
    hand_file = 'data/vids_hand_new_track.mat'
    xls_file = 'data/vidnamekey.xlsx'
    
    layer_file = 'data/vids_set_hooking_RDMs.mat'
    
    noise_ceilings = {'Visual': [0.5203, 0.5576], 'Movement': [0.7783, 0.7947], 'Goals': [0.5490, 0.5935], 'Intuitive': 0.4935}
    
    ''' Layer Analysis '''
    
#    analysis = LayerDataAnalysis(layer_file, behaviors=['visual', 'movement', 'goals'])
#    scores = analysis.cross_validation(method='reg')

#    for behavior in scores:
#        util.plot_layer_bar(scores[behavior], title='Layer cross-validation accuracies for {}'.format(behavior))

#    set_trace()
#
#    for behavior in scores:
#        reg = analysis.regression_analysis(scores[behavior], [1,2], behavior)
#        util.plot_layer_fit(reg, title='Fit for {}'.format(behavior))
#
#        #util.plot_layer_fit(scores[behavior], title='Linear Fit for {}'.format(behavior), deg=1)
#        #util.plot_layer_fit(scores[behavior], title='Quadratic Fit for {}'.format(behavior), deg=2)
#
#    set_trace()
    
    ''' Pose Analysis '''
    
    complete_pose_scores = {}
    
    #analysis = BehavioralPoseDataAnalysis(body_file, hand_file, xls_file)
    #joint_count = analysis.count_joint_frequency()
    #analysis.get_split_half_reliabilities()
    
    analysis = PoseDataAnalysis(body_file, hand_file, xls_file, model='pro', behaviors=['visual', 'movement', 'goals'])
    joint_count, rv, rj = analysis.count_joint_frequency()
    #rv = []; rj = []
    
    for model in ['avg', 'pro']:
        analysis = PoseDataAnalysis(body_file, hand_file, xls_file, model=model, removed_videos=rv, removed_joints=rj, behaviors=['visual', 'movement', 'goals'])
#        analysis = BehavioralPoseDataAnalysis(body_file, hand_file, xls_file, model=model, behaviors=['intuitive'])
        scores = analysis.cross_validation(method='reg')
        analysis.wilcoxon_test(scores)
        complete_pose_scores[model] = scores
    
    analysis.ols_reg_comparison(complete_pose_scores)
    util.plot_error_bar(complete_pose_scores, noise_ceilings=noise_ceilings, title='Comparison of Average and Procrustes distance cross-validation accuracies')
    
    '''
    # Analysis requiring unraveled frame data
    # Plots

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
