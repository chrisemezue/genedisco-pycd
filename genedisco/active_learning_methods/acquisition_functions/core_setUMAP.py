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
import umap
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


class CoreSetUMAP(BaseBatchAcquisitionFunction):
    def initialize(self,dataset_x):
        #Fit UMAP on the whole dataset
        self.embedding = umap.UMAP(metric='euclidean', n_components=16).fit(dataset_x.get_data()[0])

    def __init__(self):
        self.initialized=False
        self.embedding= None
        
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        if not self.initialized:
            self.initialize(dataset_x)
            self.initialized=True         
        
        topmost_hidden_representation = self.embedding.transform(dataset_x.subset(available_indices).get_data()[0]) 
       
        selected_hidden_representations = self.embedding.transform(dataset_x.subset(last_selected_indices).get_data()[0])
        chosen = self.select_most_distant(topmost_hidden_representation, selected_hidden_representations, batch_size)
       
        return [available_indices[idx] for idx in chosen]

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
    
  