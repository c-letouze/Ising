# Ising
A Machine Learning project on the Ising model. The main goal is to classify its phases.

## Contents

 - `README.md`: this
 - `buildsets.py` : 
 - `preprocess_CNN.py`
 
**CNN**
- `CNN.py`
- `_CNN.pkl` : CNN-preprocessed datasets

**Data_antiferro**
The antiferromagnetic samples generated by `MCSImulation/antiferromagnetic.py`. 

**Data_Mehta**
Data for the ferromagnetic samples. Downloaded from https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/

**DataSets**
Labelled (temperature and phase) datasets for the ferromagnetic and antiferromagnetic sets. They are organized by phase.

**MCSimulation**
 - `antiferromagnetism.py` generates the antiferromagnetic samples
 
 **PCA**
 - `PCA.py` : doing PCA on the ferro and antiferro datasets and classify
