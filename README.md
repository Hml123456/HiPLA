# HiPLA: An Efficient Hierarchical Graph Model for Protein–Ligand Binding Affinity Prediction



## Key Contributions

  - We propose HiPLA, a hierarchical graph framework that formulates protein-ligand binding affinity prediction as coordinated interaction modeling across multiple structural levels.

  - HiPLA jointly captures short-range 3D contacts at the binding interface and the topological organization of ligand and protein-pocket subgraphs, enabling local geometric patterns and broader structural context to be learned together.
    
  - We design a bidirectional cross-attention and global-context-conditioned gating mechanism to model protein-ligand interdependence and adaptively integrate complementary representations for affinity prediction.
  
  - Extensive experiments on multiple benchmark datasets demonstrate that HiPLA achieves competitive predictive performance and stable generalization in protein-ligand binding affinity prediction.


## Dataset
All data used in this paper are publicly available at the following locations:

- The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).

- The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org).

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

