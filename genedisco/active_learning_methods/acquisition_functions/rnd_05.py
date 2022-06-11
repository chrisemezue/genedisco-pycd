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
import scipy
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


def softmax_temperature(x, temperature=1):
    """Computes softmax probabilities from unnormalized values

    Args:

        x: array-like list of energy values.
        temperature: a positive real value.

    Returns:
        outputs: ndarray or list (dependin on x type) that is
            exp(x / temperature) / sum(exp(x / temperature)).
    """
    if isinstance(x, list):
        y = np.array(x)
    else:
        y = x
    y = np.exp(y / temperature)
    out_np = scipy.special.softmax(y)
    if any(np.isnan(out_np)):
        raise ValueError("Temperature is too extreme.")
    if isinstance(x, list):
        return [out_item for out_item in out_np]
    else:
        return out_np


class MyModel(nn.Module):
    def __init__(self,input_size):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32, 32)

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


class RND_05(BaseBatchAcquisitionFunction):
    def initialize(self,input_size):
       # initialize random model with random fixed weights
       # initialize RND
       # initialize optimizer
       self.random_model = initialize_model(input_size)

       self.RND = initialize_model(input_size)
       self.optimizer = optim.Adam(self.RND.parameters(),lr=0.01)


    def __init__(self):
        self.acquired = []
        self.initialized=False


    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:

        trained_indices = list(set(dataset_x.get_row_names())-set(available_indices))

        available_data = torch.Tensor(dataset_x.subset(available_indices).get_data()[0])
        trained_data = torch.Tensor(dataset_x.subset(available_indices).get_data()[0])

        last_selected_data = dataset_x.subset(last_selected_indices).get_data()[0]
        input_size = available_data.shape[1]

        if not self.initialized:
            self.initialize(input_size)
            self.initialized=True

        with torch.no_grad():
            input_ = torch.squeeze(self.RND(available_data))
            target_ = torch.squeeze(self.random_model(available_data))
            scores = F.mse_loss(input_,target_,reduction='none').mean(-1).detach().numpy()

        selection_probabilities = softmax_temperature(scores+1e-10, 0.5)

        numerical_selected_indices = np.random.choice(
            range(len(selection_probabilities)),
            size=batch_size,
            replace=False,
            p=selection_probabilities)
        selected_indices = [available_indices[i] for i
                            in numerical_selected_indices]

        #train on points_to_score
        self.train_on_points_to_acquire(selected_indices, dataset_x)

        #end_time = time.time()
        # import pdb; pdb.set_trace()

        return selected_indices


    def train_on_points_to_acquire(self, points_to_score, dataset_x):
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
        EPOCH = 100
        points_to_score_data = torch.Tensor(dataset_x.subset(points_to_score).get_data()[0]) #This is the training data

        if len(set(self.acquired).intersection(set(points_to_score))) > 0:
            import pdb; pdb.set_trace()

        self.acquired.extend(points_to_score)

        # Sample slightly more indices to avoid biasing towards start of dataset
        indices = torch.randperm(1024) % points_to_score_data.shape[0]
        data = points_to_score_data[indices]

        self.RND.train() #Set to train mode
        for epoch in range(EPOCH):
            self.optimizer.zero_grad()
            with torch.no_grad():
                random_labels =  self.random_model(data)
            pred_labels =  self.RND(data)
            loss = criterion(pred_labels,random_labels)
            loss.backward()
            print(f'LOSS: {loss.item()}')
            self.optimizer.step()
        self.RND.eval() # set to non-train mode.