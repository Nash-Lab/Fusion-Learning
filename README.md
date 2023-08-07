 # FusionLearning

![Alt text](TOC.png?raw=true "Title")

 Python script used to perform Forced Unfolding and Supervised Iterative Online (FUSION) learning analysis in the context of the publication entitled "Iterative Machine Learning for Classification and Discovery of Single-molecule Unfolding Trajectories from Force Spectroscopy Data".

 - [Overview](##overview)
 - [Software Requirements](##Software-Requirements)
 - [Usage](##Usage)
 - [Citation](##Citation)


 ## Overview
`FusionLearning.py` is a Python script that conducts evaluation and report of automatic screening of AFM data accordingly with FUSION learning methodology presented in "Iterative Machine Learning for Classification and Discovery of Single-molecule Unfolding Trajectories from Force Spectroscopy Data" (citation).

`environment.yml` contains the necessary information regarding the dependencies used to run `FusionLearning.py`.

`Data.zip` contains the raw data utilised in our work. Specifically, of the experimental and simulation case studies. It needs to be unzipped before being able to run `FusionLearning.py`

`Figures/` contains the figures exported from `FusionLearning.py`.

`ExportedData/` folder to export the data generated from `FusionLearning.py`. This includes the variables used and calculated (general and per each iteration) in the form of pkl files, and the neural networks trained.

## Software Requirements
`FusionLearning.py` has been tested on and Intel-based Macbook Pro (Macos 13.4.1).

To replicate our work, one should instal the libraries in `environment.yml`. We included the exact channels and builds of each library has been used. Important: using different channels, builds or versions could lead to different outcomes from the ones presented.

```
conda env create --file=environment.yml
```

Note that `Data.zip` needs to to be unzipped before being able to run `FusionLearning.py`

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

## Citation
For usage of the package and associated manuscript, please cite according to the enclosed [citation.bib]().
