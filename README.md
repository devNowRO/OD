# Scene Graph Generation And Trajectory Estemation

## General Python Setup
All of our experiments were performed on a Linux Server running Ubuntu 22.0 LTS. 


First, download and install Anaconda here:
https://www.anaconda.com/products/individual

If you are using a GPU, install the corresponding CUDA toolkit for your hardware from Nvidia here:
https://developer.nvidia.com/cuda-toolkit

Next, create a conda virtual environment running Python 3.6:
```shell
conda create --name av python=3.9
```

After setting up your environment. Activate it with the following command:

```shell
conda activate av
```

Install PyTorch to your conda virtual environment by following the instructions here for your CUDA version:
https://pytorch.org/get-started/locally/

In used Torch 1.9 and CUDA 11.8.


Next, install the PyTorch Geometric library by running the corresponding commands for your Torch and CUDA version:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Once this setup is completed, install the rest of the requirements from requirements.txt:

```shell
pip install -r requirements.txt
```

If you want to visualize the extracted scene-graphs, as is done in Use-Case 1, you can either use the networkx or pydot/graphviz APIs. Typically graphviz works better so our code defaults to graphviz. In order to render these graphs, you must have [graphviz](https://www.graphviz.org/download/) installed on your system along with the corresponding python package installed as follows:
```
conda install -c anaconda graphviz
```

If you want to use our CARLA dataset generation tools then you need to have the CARLA simulator and Python API installed as described here:
https://github.com/carla-simulator/carla/releases

https://carla.readthedocs.io/en/latest/start_quickstart/


To perform image scene-graph extraction, you must first install Detectron2 by following these instructions:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

---
## Usage Examples
### Setup and Dataset
Before running any of the use cases, please download the use_case_data directory from the following link and place it into the examples directory.
https://drive.google.com/drive/folders/1zjoixga_S8Ba7khCLBos6nhEhe1UWnv7?usp=sharing



### Use Case 1: Converting an Ego-Centric Observation (Image) into a Scene-Graph
In this use case, we demonstrate how to  extract road scenegraphs from a driving clip. In the sample script examples/scene_graph_generation.py, roadscene2vec first takes in the use_case_1_scenegraph_extraction_config.yaml config file. This file specifies the location of the data from which to extract scenegraphs from along with the various relations and actors to include in each scenegraph . A. 

To run this use case, cd into the examples folder and run the corresponding module. 

```shell
$ cd examples
$ python scene_graph_generation.py
```
# Trajectory Estemation

## Training the Model

Run the following command to train the position estimation model:

```bash
python trainingPipiline.py 

```
---

## Position Estimation (Trajectory Inference)

After training, run trajectory-based position estimation:

```bash
python trajectory.py
```



NOTE: If you would like to use a new image dataset, we recommend generating a new birds-eye view transformation (bev.json) using the instructions provided here:


# Trajectory Estemation

## Training the Model

Run the following command to train the position estimation model:

```bash
python trainingPipiline.py 

```
---

## Position Estimation (Trajectory Inference)

After training, run trajectory-based position estimation:

```bash
python trajectory.py
```