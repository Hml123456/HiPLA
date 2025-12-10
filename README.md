# HiPLA:Efficient Protein-Ligand Binding Affinity Prediction via Hierarchical Graph Interaction and Global-Local Attention Fusion

## Overview of the Architecture

![Overview of the Architecture](./assets/CheapNet_Overview.png)

## Key Contributions

  - 11111111111111111111111111111111111111111111111111111.
  - 2222222222222222222222222222222222222222222222222222222222.
  - 33333333333333333333333333333333333333333333333333333333333333333.


## Dataset
All data used in this paper are publicly available at the following locations:

The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).

The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org).

## Requirements  
dgl==0.9.0  
networkx==2.5  
numpy==1.19.2  
pandas==1.1.5  
pymol==0.1.0  
rdkit==2022.3.5  
scikit_learn==1.1.2  
scipy==1.5.2  
torch==1.10.2  
tqdm==4.63.0  
openbabel==3.3.1 (conda install -c conda-forge openbabel)  

## Descriptions of Folders and Files
- **`./data`:** Includes details regarding multiple datasets. Please download and structure the preprocessed data as specified.
- **`./config`:** Parameters used in HiPLA.
- **`./log`:** Logger.
- **`./model`:** Contains model checkpoints and training records.
- **Scripts and Implementations:** Various Python files implementing models, preprocessing, training, and testing.

## Step-by-step Running

### 1. Process Raw Data
- Download the preprocessed datasets and organize them in the `./data` folder.
  - `python preprocessing.py`
  - `python dataset_HiPLA.py`

### 2. Model Training And Testing
- After processing the dataset
  - `python train.py`

