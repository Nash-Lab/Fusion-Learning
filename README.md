 # FusionLearning

![Alt text](TOC.png?raw=true "Title")

 Python script used to perform Forced Unfolding and Supervised Iterative ONline (FUSION) learning analysis in the context of the publication entitled "Iterative Machine Learning for Classification and Discovery of Single-molecule Unfolding Trajectories from Force Spectroscopy Data" (--> [preprint](https://doi.org/10.1101/2023.08.08.552253)).

 - [Overview](##overview)
 - [Software Requirements](##Software-Requirements)
 - [Data Requirements](##Data-Requirements)
 - [Usage](##Usage)
 - [Citation](##Citation)


 ## Overview
 ### Script Information

 The `FusionLearning.py` script is designed to work with the provided npy files, which include "Simulation_train", "Simulation_test", or "Experimental" datasets. These files contain various arrays:

 `x_raw`: This array, with dimensions (n_curves, n_time_steps, parameters), contains raw data. In our specific case, the parameters consist of time (1), force (2), tip-position (3), and contour length. While this array is not directly used in the process, it is essential to be present in the form of a (n_curves, 1) array, which can contain random numbers.

 `y_raw`: This array, with dimensions (n_curves, 1), contains labels for each trace. A label of 0 denotes bad curves, while labels 1 to 3 represent pathways (good curves).

 `x_1D_processed`: With dimensions (n_curves, 600, 1), this array contains processed data, as detailed in the Supporting Information section of our paper. Similar to x_raw, it is not directly used but should be present in the form of a (n_curves, 1) array, which can also consist of random numbers.

`x_2D_processed`: These are images generated from x_1D_processed, with dimensions (n_curves, 200, 200, 1). Note that the image dimensions are not fixed at 200x200, as each image is automatically resized to (224x224x3) for processing with DenseNet.

The `FusionLearning.py` script begins by loading the necessary dependencies, functions, and data. It processes this data by performing feature extraction using DenseNet [[1](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121), [2](https://arxiv.org/abs/1608.06993)]. Subsequently, it calculates a distance matrix in a scalable manner using Dask.

The iterative procedure is then initialized. During each iteration, a specific number of curves, controlled by the --n_iter_new_curves parameter, is used added to the train/validation subset and used to train a binary classifier. This classifier is employed to order the remaining curves based on its output.

Throughout each iteration, the script provides information about the composition of the screened, unscreened, and test sets via console output. An example of this output is shown below.

```
data   screened: 4 --> bad curves: 2; P1: 2; P2: 0; P3: 0
data unscreened: 2783 --> bad curves: 1998; P1: 445; P2: 340; P3: 0
data (test set): 3000 --> bad curves: 0; P1: 469; P2: 67; P3: 7
```

During specific iterations (typically after processing 20, 500, and 1000 traces), the triplet loss based neural network is trained and evaluated on the provided test set. The embedding information generated (layer 2 and 3) with such neural network, as well as the number of good curves found (layer 1) are shown in a figure generated automatically (with option --renderfig True). Please note that rendering the figure may require LaTeX to be installed on your system. If you encounter issues related to LaTeX, we recommend deactivating the figure rendering.

To test our methodology with your own data, you can use two files named "Simulation_train.npy" and "Simulation_test.npy," which should contain the arrays described earlier in the documentation. Additionally, you may consider adjusting certain variables that we have provided in the code, such as the shape of the neural network's final layer, the number of curves processed per iteration, the threshold for training the neural network, and others, to better suit your specific requirements.

For more details, please refer to the code and to the supporting information of our paper, or contact us directly.


 ### Provided Files
`FusionLearning.py` is a Python script that conducts evaluation and report of automatic screening of AFM data accordingly with FUSION learning methodology presented in "Iterative Machine Learning for Classification and Discovery of Single-molecule Unfolding Trajectories from Force Spectroscopy Data".

`environment.yml` contains the necessary information regarding the dependencies used to run `FusionLearning.py`.

`Data.zip` (available at https://doi.org/10.5281/zenodo.8224236) contains the raw data utilised in our work. Specifically, of the experimental and simulation case studies. It needs to be downloaded separately and unzipped before being able to run `FusionLearning.py`

`Figures/` contains the figures exported from `FusionLearning.py`.

`ExportedData/` folder to export the data generated from `FusionLearning.py`. This includes the variables used and calculated (general and per each iteration) in the form of pkl files, and the neural networks trained.

## Software Requirements
### MacOS
`FusionLearning.py` has been tested on and Intel-based Macbook Pro (Macos 13.4.1).

To replicate our work, one should instal the libraries in `environment.yml`. We included the exact channels and builds of each library has been used. Important: using different channels, builds or versions could lead to different outcomes from the ones presented.

```
conda env create --file=environment.yml
```

Note that `Data.zip` needs to to be unzipped before being able to run `FusionLearning.py`

### Windows / Linux
As previously stated, `FusionLearning.py` has been tested on MacOS. If you are interested in testing it on a Windows or Linux machine you could use the following commands to create a compatible environment.
```
conda create -n FUSION_Leanrning python=3.8.3
conda activate FUSION_Leanrning
pip install tensorflow==2.6
pip install protobuf==3.20
pip install tensorflow_addons==0.14
pip install keras==2.6
pip install scikit-learn==0.23.1
pip install matplotlib==3.3.1
pip install dask==2021.7.0
```
Please note that, due to the unavailability of some dependencies utilised during the generation of random numbers on MacOS (weights initialisation and shuffling), the output might slightly differ from the one presented in our paper. This is particularly important in our case due to the iterative nature of the presented algorithm, where even small changes in the early iterations may lead to a "butterfly effect" in the latter ones.

## Data Requirements
The raw data used to run `FusionLearning.py` are available free of charge at https://doi.org/10.5281/zenodo.8224236. The linked archive should be downloaded, unziped and stored in the same directory of `FusionLearning.py`. Note that the numpy files has to be contained in a folder called `Data` (`./Data/Experiment.npz`, `./Data/Simulation_test.npz` and `./Data/Simulation_train.npz`).

## Usage
To use `FusionLearning.py` one could load the appropriate conda environment and run the script in a terminal as follow:
```
conda activate FUSION_learning
python3 ./FusionLearning.py -d experiment
```
or
```
conda activate FUSION_learning
python3 ./FusionLearning.py -d simulation
```
For an extended overview on the available parameters one could run:
```
python3 ./FusionLearning.py -h
```
If you are experiencing problems in running our script due to latex (especially with Windows), please consider to use the default option (--renderfig False).

## Citation
For usage of the package and associated manuscript, please cite according to the enclosed [citation.bib](citation.bib).
