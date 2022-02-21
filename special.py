import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

class RandomBatchAcquisitionFunction(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr], 
                 last_selected_indices: List[AnyStr] = None, 
                 model: AbstractBaseModel = None,
                 temperature: float = 0.9,
                 ) -> List:
        selected = np.random.choice(available_indices, size=batch_size, replace=False)
        return selected