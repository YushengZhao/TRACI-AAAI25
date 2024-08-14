# TRACI: A Data-centric Approach for Multi-Domain Generalization on Graphs

## Experiment Environment
* Operating System: Linux (Ubuntu 18.04.6 LTS)
* GPU: NVIDIA A40
* Major Package Requirements (details in requirements.txt)
  * python 3.10
  * pytorch 1.12
  * torch-cluster 1.6
  * torch-scatter 2.0
  * torch-sparse 0.6

## Instructions
### Step 1: Install required packages
```
conda create -n traci
conda activate traci
pip install -r requirements.txt
```

### Step 2: Run the script
```
python run.py --experiment citation_or_protein --gpu your_gpu_id --method traci --source-names the_list_of_source_names --target-name the_target_name
```
