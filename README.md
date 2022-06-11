## GeneDisco competition 1st place submission (team pycd)

![image](https://user-images.githubusercontent.com/36100251/172914641-6c1aefff-e72d-4a7d-a918-d95968768edd.png)
[We won the challenge🤩!](https://twitter.com/DariaYasafova/status/1520137801894969344)


We used [Random Network Distilation model](genedisco/active_learning_methods/acquisition_functions/rnd_05.py) for our final submission. Please refer to the [paper](https://arxiv.org/abs/1810.12894) to learn more about the method.


# GeneDisco: A benchmark for active learning in drug discovery

![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-1.0.0-blue)

In vitro cellular experimentation with genetic interventions, using for example CRISPR technologies, is an essential 
step in early-stage drug discovery and target validation that serves to assess initial hypotheses about causal 
associations between biological mechanisms and disease pathologies. With billions of potential hypotheses to test, 
the experimental design space for in vitro genetic experiments is extremely vast, and the available experimental 
capacity - even at the largest research institutions in the world - pales in relation to the size of this biological 
hypothesis space. 

[GeneDisco (published at ICLR-22)](https://arxiv.org/abs/2110.11875) is a benchmark suite for evaluating active learning algorithms for experimental design in drug discovery. 
GeneDisco contains a curated set of multiple publicly available experimental data sets as well as open-source i
mplementations of state-of-the-art active learning policies for experimental design and exploration.

## GeneDisco ICLR-22 Challenge

### License

[License](LICENSE.txt)


Contribution from  **pycd** team:


the following files are under [our MIT License](https://github.com/chrisemezue/genedisco-pycd/blob/master/LICENSE_MIT.md):
```
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/core_set.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/core_set2.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/core_setUMAP.py 
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/ensemble_rnd.py 
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/rnd.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/uncertainty_acquisition.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/uncertainty_acquisition_03.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/uncertainty_acquisition_05.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/uncertainty_acquisition_07.py
genedisco-pycd/genedisco/active_learning_methods/acquisition_functions/uncertainty_acquisition_10.py
genedisco-pycd/genedisco/visualization/visualization_2.ipynb
genedisco-pycd/genedisco/visualization/viz.py
genedisco-pycd/genedisco/visualization/viz_utils.py
```

Please note that it is possible to open genedisco-pycd/genedisco/visualization/visualization_2.ipynb, even though it's too large to view it on github.
To do that, open the notebook as a raw file, copy its content, and save it on your local machine as an .ipynb file.
The notebook contains some of our latest visualization comparisons that we used to choose the best performing acquisition function among those we tried.


