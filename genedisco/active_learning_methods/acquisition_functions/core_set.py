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
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


class CoreSet(BaseBatchAcquisitionFunction):
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        topmost_hidden_representation = last_model.get_embedding(dataset_x.subset(available_indices)).numpy()
        selected_hidden_representations = last_model.get_embedding(dataset_x.subset(last_selected_indices)).numpy()
        chosen = self.select_most_distant(topmost_hidden_representation, selected_hidden_representations, batch_size)
        chosen2 = self.select_most_distant_with_kde(topmost_hidden_representation, selected_hidden_representations, batch_size)
        #import pdb; pdb.set_trace()  
        return [available_indices[idx] for idx in chosen2] #change to `chosen` to use the default one - non-KDE   

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
    
    def select_most_distant_with_kde(self, options, previously_selected, num_samples): #This one uses Kernel Density
        
            num_options, num_selected = len(options), len(previously_selected)
            if num_selected == 0:
                min_dist = np.tile(float("inf"), num_options)
            else:
                # Two options:
                # 1. Use prevously_selected for fitting and `score_samples` on options (currently using this).
                # 2. Use prevously_selected+options for fitting and `score_samples`` on options.
                kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(previously_selected)
                log_likelihood_options = kde.score_samples(options) #log-likelihood of each sample in options under the model: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.score_samples 
                min_dist = -1 * log_likelihood_options # (negative_log_likelihood_options). This will make the concept of min and max the same with when using pairwise distance
             
            indices = []
            for i in range(num_samples):
                idx = min_dist.argmax()
                kde_one = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(options[[idx], :]) # I question doing KDE on one data point. 
                new_log_likelihood_options = kde_one.score_samples(options)
                dist_new_ctr = -1 * new_log_likelihood_options
                for j in range(num_options):
                    min_dist[j] = min(min_dist[j], dist_new_ctr[j])
                indices.append(idx)
            return indices
