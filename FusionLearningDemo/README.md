 # FusionLearningDemo

This Demo aims to showcase Fusion Learning functionalities, necessary inputs and produced outputs on a simple example.

This Demo features a class, `FusionLearning`, designed to simulate a possible implementation of asynchronous operation within a Docker container environment. It is initialised with AFM trace images, from which it extracts features using the DenseNet121 neural network. Subsequently, it computes a distance matrix for each data point within an experiment. The `training_and_exporting_results` method can then be invoked with a pandas DataFrame. This DataFrame should include curve IDs, distance matrix indexes (idx), indicators of whether an entry has been labelled (labelled), and the labels themselves (label). This method yields scores for various Fusion Learning layers, namely:

* Layer 1 for classification
* Layer 2 for embedding
* Layer 3 for posterior covariance

The parameters train_embedding and output_covariance allow for the control over the training and output of the latter two layers, respectively.
