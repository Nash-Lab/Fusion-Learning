conda create -y --name FUSION_learning
conda activate FUSION_learning
conda install -y pip=21.1.2=py38hecd8cb5_0
conda install -y -c anaconda python=3.8.3=h26836e1_2
conda install -y -c conda-forge tensorflow=2.6.0=py38h52b2510_1                          
conda install -y scikit-learn=0.23.1=py38h603561c_0
conda install -y -c anaconda matplotlib=3.3.1 
conda install -y -c anaconda dask=2021.7.0=pyhd3eb1b0_0
pip install tensorflow-addons==0.14.0
pip install tensorflow==2.6.0 tensorflow-addons