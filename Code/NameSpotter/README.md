# NameSpotter
**NameSpotter** is designed to automatically identify abnormal method names with graph neural networks. Besides the method names, we also leverage their POS tags and parameter information as the input. **NameSpotter** is equipped with three graph neural networks for extracting semantic and syntactic features of the inputs separately. The outputs of the three graphs are hierarchically pooled and aggregated, resulting in a method name feature representation for the final identification. Our evaluation results on 4,327 real-world method names suggest that **NameSpotter** performs effectively in identifying abnormal method names with an accuracy of 91.9%, outperforming the state-of-the-art approach by 21.5%. 
<p align="center"><img src="Approach_version3.png" alt="logo" width="800px" />

## Environment  
We implement NameSpotter with **PyTorch**, 
### Torch Version:
- Python 3.6.9
- Pytorch 1.2

and the other required packages can be found in **requirements.txt**.



## Quick Start
(1) Clone the repository from GitHub:
```
git clone https://github.com/AnonymousAccountSE/NameSpotter_OnlineRepos.git
```

(2) Configure the python and PyTorch environment and set up the environments by:
```
pip install -r requirements.txt
```

(3) Switch to the NameSpotter directory:
```
cd Code/NameSpotter
```
(4) Preprocess the data and get them prepared for training by:
```
cd preprocess
python proprocess.py
```
(5) Then, NameSpotter should work on our manually labelled dataset by:
```
cd PyTorch
python train.py
```
In addition, you can choose the specific GPU by:
```
python train.py --gpu 2
```

## Tuning NameSpotter

The search space of all the hyperparameters can be found in search_space.txt. We follow the grid search strategy to perform the parameter tuning.




