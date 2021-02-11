import sys
import numpy as np
import scipy.io
import mat73
import time

import analysis_util as util
import analysis as analysis

from scipy.stats import zscore, pearsonr, percentileofscore, spearmanr
from scipy.spatial.distance import pdist, squareform

from pdb import set_trace

def triu(array):

    '''
    Extract upper triangle of a square array
    '''

    return array[np.triu_indices(len(array),k=1)]

def unpad(array):

    '''
    Unpads a brain patterns array
    '''

    inds = np.ravel(np.argwhere(~np.isnan(array[0])))
    to_return = []
    for a in array:
        to_return.append(a[inds])
    return np.array(to_return)
    
def load_layer_data(file):

    '''
    Loads layer data. Returns list of length 55 with averaged layer output
    data for all layers.
    '''
    
    layers = scipy.io.loadmat(file)
    numerical_layers = []
    
    # Delete extra dictionary keys
    del layers['__header__']
    del layers['__version__']
    del layers['__globals__']
    
    for layer in layers:
        #l = layer.split('_')[1]
        numerical_layers.append(layers[layer][0])
    
    return numerical_layers

class BrainData():

    def __init__(self, brain_file, brain_patterns):
        
        print('Loading brain data files... may take a while (~2 minutes)')
        start_time = time.time()
        
        data = scipy.io.loadmat(brain_file)
        self.data = np.transpose(data['searchlightRDMs'], (2,0,1))
        self.voxel_IDs = np.ravel(data['grayMatterVoxelIdx'])
        
        self.patterns = np.transpose(mat73.loadmat(brain_patterns)['searchlightBrainPatterns'], (2,0,1))
        
        print("--- Loading data completed in %s minutes ---" % ((time.time() - start_time)/60))
        
    def correlate_brain(self, target):
    
        '''
        Computes correlation between all BrainData distance vectors and a
        given array distance vector. Returns 1d vector of size len(data)
        '''
        
        print('Performing brain-pose correlations... this will be quick')
        
        brain_correlations = [pearsonr(triu(vox), target)[0] for vox in self.data]
        return np.array(brain_correlations)
        
    def voxwise_perm_tests(self, correlations, target, k):
        
        '''
        For every voxel, shuffle brain brain data and creates new RDM.
        Compare this new RDM with target and repeat k times. Voxel-layer
        comparison is valid if correlation is in 99th percentile of random
        correlations. Returns list of 99th percentile cutoffs for each voxel,
        and the indices of valid voxels (as determined by this test).
        '''
        
        print('Performing voxel permutation tests... *will* take a while (~2 hours)')
        start_time = time.time()
        
        real_indices = []; cutoffs = []
        for i,p in enumerate(self.patterns):
        
            # outputs progress
            if i%(int(len(self.patterns)/20)) is 0:
                print('{}% complete'.format(int(i/len(self.patterns)*100)))
        
            pat = unpad(p)
            
            #shuffled_corrs = [pearsonr(pdist(pat[np.random.permutation(pat.shape[0])], metric='sqeuclidean'), target)[0] for j in range(k)]
            
            # shuffles brain pattern, creates new RDM based on shuffled
            # and then correlates new RDM with target. Repeats k times
            
            shuffled_corrs = []
            for j in range(k):
                shuffled_corrs.append(spearmanr(pdist(pat[np.random.permutation(pat.shape[0])], metric='sqeuclidean'), target, nan_policy='omit')[0])
            
            # if the correlation calculated earlier is above the 99th percentile
            # then the voxel correlation is valid
            cutoff = np.percentile(shuffled_corrs, 99)
            cutoffs.append(cutoff)
            if correlations[i] > cutoff:
                real_indices.append(self.voxel_IDs[i])
#            if percentileofscore(shuffled_corrs, correlations[i]) >= 0.99:
            
        print("--- Voxel-wise tests completed in {} minutes ---".format((time.time() - start_time)/60))
        return real_indices, cutoffs

if __name__ == '__main__':
    
    brain_file = 'data/RSAData-formatted-set1.mat'
    brain_patterns = 'data/RSAData-formatted-BrainPatterns-set1.mat'
    body_file = 'data/vids_body_new_track.mat'
    hand_file = 'data/vids_hand_new_track.mat'
    xls_file = 'data/vidnamekey.xlsx'
    
    # Create new brain analysis
    brain_data = BrainData(brain_file, brain_patterns)
    
    # Extract layer data
    layers_of_interest = [19, 26, 33, 40, 47, 54] # final layer of each stream
    layer_file = 'data/hooking_openpose_RDMs.mat'
    layer_data = load_layer_data(layer_file)
    
    '''
    # Retrieve target RDM from pose analysis (average)
    pose = analysis.PoseDataAnalysis(body_file, hand_file, xls_file, model='avg')
    target_RDM = pdist(pose.data.T.values, metric='sqeuclidean')
    target_RDM = zscore(target_RDM, nan_policy='omit')
    #target_RDM = squareform(target_RDM)
    '''
    
    # Perform brain analysis operation for each layer
    for i, layer in enumerate(layers_of_interest, 1):
        
        print('Beginning analysis of layer {} ({}/{})'.format(layer, i, len(layers_of_interest)))
        
        target_RDM = zscore(layer_data[layer], nan_policy='omit')
        
        # Correlate
        brain_layer_corrs = brain_data.correlate_brain(target_RDM)
        
        # Voxel-Wise corrections
        real_indices, cutoffs = brain_data.voxwise_perm_tests(brain_layer_corrs, target_RDM, 100)
    
        # Write to matlab file
        to_matlab = {'grayMatterVoxelIdx': brain_data.voxel_IDs, 'brainLayerCorrelations': brain_layer_corrs, 'voxelWiseSurvivingIds': real_indices, 'cutoffs': cutoffs}
    
        scipy.io.savemat('data/Brain-Layer{}-voxelwise-set1.mat'.format(layer), to_matlab)
        
    print('Program complete!')
