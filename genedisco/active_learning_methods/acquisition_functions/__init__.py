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

from .random_acquisition_function import RandomBatchAcquisitionFunction
from .margin_sampling_acquisition import MarginSamplingAcquisition
from .kmeans import KMeans
from .core_set import CoreSet
from .core_set2 import CoreSetKDE
from .core_setUMAP import CoreSetUMAP
from .badge_sampling import BadgeSampling
from .adversarial_bim import AdversarialBIM
from .uncertainty_acquisition import TopUncertainAcquisition, SoftUncertainAcquisition
from .rnd import RND
from .rnd_05 import RND_05
from .uncertainty_acquisition_03 import SoftUncertainAcquisition_03
from .uncertainty_acquisition_05 import SoftUncertainAcquisition_05
from .uncertainty_acquisition_07 import SoftUncertainAcquisition_07
from .uncertainty_acquisition_10 import SoftUncertainAcquisition_10
