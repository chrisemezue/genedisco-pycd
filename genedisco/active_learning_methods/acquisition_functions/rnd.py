"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

class MyModel(nn.Module):
    def __init__(self,input_size):
        super(MyModel, self).__init__()
        
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,1)
        
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x) 
        x = self.linear2(x)
        x = F.relu(x) 
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x


def initialize_model(input_size):
    my_random_model = MyModel(input_size)
    return my_random_model 


class RND(BaseBatchAcquisitionFunction):
    def initialize(self,input_size):
       # initialize random model with random fixed weights
       # initialize RND
       # initialize optimizer
       self.random_model = initialize_model(input_size)  
       self.RND = initialize_model(input_size) 
       self.optimizer = optim.Adam(self.RND.parameters(),lr=0.0001)


    def __init__(self):
        self.initialized=False
      
        
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
               
      
        available_data = torch.Tensor(dataset_x.subset(available_indices).get_data()[0])
        last_selected_data = dataset_x.subset(last_selected_indices).get_data()[0]
        input_size = available_data.shape[1]
       

        if not self.initialized:
            self.initialize(input_size)
            self.initialized=True         
        
        input_ = torch.squeeze(self.RND(available_data))
        target_ = torch.squeeze(self.random_model(available_data))
        scores = F.mse_loss(input_,target_,reduction='none')
        _,scores_sorted = torch.sort(scores, descending=True) 

        scores_sorted = scores_sorted.numpy().tolist()[:batch_size]

        points_to_score = [available_indices[idx] for idx in scores_sorted]

        #train on points_to_score
        self.train_on_points_to_acquire(points_to_score,dataset_x)      
        
        
        return points_to_score


    def train_on_points_to_acquire(self,points_to_score,dataset_x):
        """
            Given: random_model and RND and points_to_acquire
        for epoch in EPOCH:
            Get label_of_points_to_acquire from random_model(points_to_acquire)
            Get  label_of_points_to_acquire from RND(points_to_acquire)
            Get loss: MSE(label_of_points_to_acquire,label_of_points_to_acquire) loss is by sum/mean
            do backward step
            optimizer.step()
            
        """
        criterion = nn.MSELoss()
        EPOCH=20
        points_to_score_data = torch.Tensor(dataset_x.subset(points_to_score).get_data()[0]) #This is the training data
        self.RND.train() #Set to train mode
        for epoch in range(EPOCH):

            self.optimizer.zero_grad()
            random_labels =  self.random_model(points_to_score_data)
            pred_labels =  self.RND(points_to_score_data)
            loss = criterion(pred_labels,random_labels)
            loss.backward()
            print(f'LOSS: {loss.item()}')
            self.optimizer.step()
        import pdb; pdb.set_trace()


        self.RND.eval() # set to non-train mode.    


    def select_most_distant(self, options, previously_selected, num_samples):
       
        num_options, num_selected = len(options), len(previously_selected)
        if num_selected == 0:
            min_dist = np.tile(float("inf"), num_options)
        else:
            dist_ctr = pairwise_distances(options, previously_selected)
            min_dist = np.amin(dist_ctr, axis=1)
          
        indices = []
        for i in range(num_samples):
            idx = min_dist.argmax()
            dist_new_ctr = pairwise_distances(options, options[[idx], :])
            for j in range(num_options):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
            indices.append(idx)
        return indices
    
  