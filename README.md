# Replication package for NameSpotter.
 ***
 ## Content of the replication package

 /Code: The implementation of NameSpotter with detailed replication steps.

 /Data: Publicly available method names dataset, which is labeled by five evaluators.

 /LiveStudy: The results of the live study conducted on developers, with addresses of pull requests provided.
  ***
 ## How to replicate the evaluation of NameSpotter?
 
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
The detailed replication instructions can be found in /Code/NameSpotter/README.md. 
 ***