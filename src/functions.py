'''
Various functions used in code

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from pdb import set_trace
import numpy as np

class Hook():

    #def __init__(self, module, model, data, layer_index):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook)
        #self.layer_index = layer_index
        
    def hook(self, module, input, output):
        #print('here at' + str(self.layer_index))
        #self.input = input
        #self.output = output
        self.output = output.clone().detach().requires_grad_(True)
    
    def close(self):
        self.hook.remove()
        
def get_layer_names(layers):

    layer_names = []
    layer_ind = 0
    for i, layer in enumerate(layers):
        layer_name = str(layer).split('(')[0]
        layer_names.append([i, str(i+1) + '-' + layer_name + '-' + str(sum(layer_name in string[1].split('-')[1] for string in layer_names) + 1)])
        #layer_ind = layer_ind + 1
    return layer_names
