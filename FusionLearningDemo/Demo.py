#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 11:25:56 2024

@author: vdoffini
"""
#%% Import Libraries
import tensorflow as tf #tf.__version__ == 2.6.0
from tensorflow_addons.losses import TripletHardLoss #tfa.__version__ == 0.14.0
# tf.compat.v1.disable_eager_execution()# necessary to disable a tensorflow warning about retracing in a loop. This could be triggered by custom metric, which in this case might be TriplettLoss. See https://stackoverflow.com/questions/58814130/tensorflow-2-0-custom-keras-metric-caused-tf-function-retracing-warning
import numpy as np #np.__version__ == 1.19.2
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import dask.array as da #dask.__version__ == 2021.07.0
from dask.diagnostics import ProgressBar
import pandas as pd
from scipy.linalg import cho_factor,cho_solve

#%% Functions and FusionLearning Class
def load_images_labels_from_npz(file_path = './AFM_data.npz'):
    """
    Load images and labels from a .npz file.

    This function loads 2D processed images and their corresponding labels from a specified .npz file. The .npz file
    is expected to contain two arrays: 'x_2D_processed' for the images and 'y_raw' for the labels. The labels are
    reshaped into a two-dimensional array with a single column.

    Parameters
    ----------
    file_path : str, optional
        The path to the .npz file containing the images and labels. Defaults to './AFM_data.npz'.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two elements:
        - The first element is a numpy array of the 2D processed images.
        - The second element is a numpy array of the labels, reshaped into a two-dimensional array with a single column.

    Example
    -------
    >>> images, labels = load_images_labels_from_npz('./AFM_data.npz')
    >>> print(images.shape)
    >>> print(labels.shape)

    Note
    ----
    The function uses `numpy.load` to load the data from the .npz file. Ensure that the file exists at the specified
    path and contains the required arrays with the exact names 'x_2D_processed' and 'y_raw'.
    """

    with np.load(file_path) as f:
        images = f['x_2D_processed']
        total_labels = f['y_raw'].reshape(-1,1)
    return (images,total_labels)

def get_inputs_4_docker_cont(total_labels,selected_indexes):
    """
    Prepare inputs for Docker containers by organizing selected and unselected indices with labels.

    This function iterates over an images collection (implicitly referenced) and creates a dictionary that maps each
    image to its index, a flag indicating whether it was selected, and its label (if selected, otherwise np.nan).
    It then converts this dictionary into a pandas DataFrame, with each image represented by a row. The DataFrame is
    structured to easily identify which images have been selected and their labels, making it suitable for use in
    Docker container inputs or other processing workflows.

    Parameters
    ----------
    total_labels : numpy.ndarray
        An array containing the labels for the images. The labels should match the order of the images.
    selected_indexes : list or array-like
        A list or array-like object containing the indexes of the selected images.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to an image. Columns include:
        - 'idx': The index of the image.
        - 'labelled': A flag indicating whether the image was selected (1) or not (0).
        - 'label': The label of the image if selected, otherwise np.nan.

    Example
    -------
    >>> total_labels = np.array([...])
    >>> selected_indexes = [0, 2, 5]
    >>> inputs_df = get_inputs_4_docker_cont(total_labels, selected_indexes)
    >>> print(inputs_df.head())

    Note
    ----
    This function implicitly assumes access to a variable named 'images' that determines the range of iteration.
    Ensure that 'images' is defined in the scope where this function is called. The function uses `pandas` for
    DataFrame creation and manipulation, so ensure that pandas is installed and imported as `pd` in your environment.
    """

    d = {}
    for i in range(len(images)):
        if i in selected_indexes:
            d[f'AFM_{i:04d}'] = {'idx':i,
                                 'labelled':1,
                                 'label':total_labels[i][0],
                                 }
        else:
            d[f'AFM_{i:04d}'] = {'idx':i,
                                 'labelled':0,
                                 'label':np.nan,
                                 }
    out = pd.DataFrame(d).T.convert_dtypes()
    return out

class FusionLearning():
    def __init__(self,images):
        self.images = images[...,np.newaxis]
        self.features = self.feature_extraction()
        self.distance_matrix = self.calculate_distance_matrix(2)

        self.classifier = LogisticRegression(solver='liblinear')#solver='liblinear' is necessary to avoid the error "AttributeError: 'str' object has no attribute 'decode'" --> see https://stackoverflow.com/questions/65682019/attributeerror-str-object-has-no-attribute-decode-in-fitting-logistic-regre
        self.embedder = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
        self.embedder.compile(loss=TripletHardLoss())

        self.optimal_hyperparameters = None
        
        #outputs
        self.score = None
        self.embedding = None
        self.covariance = None
    

    def feature_extraction(self):
        """
        Extract features from images using the DenseNet121 model.

        This method applies the DenseNet121 model, a deep convolutional neural network, for feature extraction on the
        instance's images. The images are first preprocessed by inverting their colors, duplicating the color channels
        to match the input requirements of DenseNet121 (converting grayscale images to RGB by repetition), resizing to
        224x224 pixels, and applying the necessary DenseNet-specific preprocessing steps. The output is the extracted
        features from the second-to-last layer of the DenseNet121 model, providing a rich representation of the images.

        The method assumes that `self.images` contains the images to process, stored as a numpy array. These images should
        be in grayscale and have their pixel values in the range [0, 255].

        Returns
        -------
        numpy.ndarray
            An array containing the extracted features for each image. Each row in the array corresponds to the features
            of one image.

        Example
        -------
        Assuming an instance `instance` of a class with `feature_extraction` method and `self.images` initialized:

        >>> features = instance.feature_extraction()
        >>> print(features.shape)

        Note
        ----
        This method requires TensorFlow and the TensorFlow Keras applications module to be installed. The DenseNet121 model
        is used for feature extraction, and its weights are automatically downloaded the first time this method is called.
        """

        model_temp = tf.keras.applications.DenseNet121()
        feature_extractor = tf.keras.models.Model(inputs=model_temp.input,outputs=model_temp.layers[-2].output)
        images_preprocessed = tf.keras.applications.densenet.preprocess_input(tf.image.resize(255-np.repeat(self.images,3,axis=-1).astype(np.float32),(224,224),'nearest'))
        features = feature_extractor.predict(images_preprocessed)
        return features

    def calculate_distance_matrix(self,l):
        """
        Calculate the L-norm distance matrix between features.

        This method computes the distance matrix for the features stored in `self.features` using the L-norm specified by
        the parameter `l`. It utilizes Dask for parallel computation, making it efficient for large sets of features.
        The method initially calculates the difference between each pair of features, then applies the L-norm over these
        differences to obtain the distance matrix. The computed distances are raised to the power of `l` to complete the
        calculation.

        Parameters
        ----------
        l : int
            The order of the norm to be used in distance calculations. For example, `l=2` calculates the Euclidean
            distance.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array representing the distance matrix, where each element (i, j) is the distance between the i-th
            and j-th feature vectors according to the L-norm.

        Example
        -------
        Assuming an instance `instance` of a class with the `calculate_distance_matrix` method and `self.features` properly
        initialized:

        >>> distance_matrix = instance.calculate_distance_matrix(l=2)
        >>> print(distance_matrix.shape)

        Note
        ----
        This method assumes that `self.features` is a numpy array containing the feature vectors for which the distance
        matrix is to be computed. It also requires Dask for parallel computation, so ensure that Dask is installed and
        imported as `da`. The choice of `l` allows for flexibility in the distance metric, accommodating various norms.
        """

        x1 = self.features
        x2 = self.features
        # ProgressBar().register()
        a1 = da.array(x1,dtype=np.float32)
        a1 = a1.reshape((x1.shape[0],1,x1.shape[1]))
        a2 = da.array(x2,dtype=np.float32)
        a2 = a2.reshape((1,x2.shape[0],x2.shape[1]))
        a3 = a1-a2
        chunk_size = a3.rechunk().chunksize
        
        a1 = da.array(x1,dtype=np.float32).rechunk((chunk_size[0],chunk_size[2]))
        a1 = a1.reshape((x1.shape[0],1,x1.shape[1]))
        
        a2 = da.array(x2,dtype=np.float32).rechunk((chunk_size[1],chunk_size[2]))
        a2 = a2.reshape((1,x2.shape[0],x2.shape[1]))
        
        a3 = a1-a2

        a4 = da.linalg.norm(a3, ord=l, axis=-1)
        distance_matrix = a4.compute()
        return distance_matrix**l
    
    def calculate_kernel_matrix(self,distance_matrix,scale):
        """
        Calculate the distance matrix for the instance's features using Dask for efficient computation.

        This method computes the pairwise distances between feature vectors stored in `self.features` using the norm specified
        by the parameter `l`. It leverages Dask arrays for handling potentially large datasets that may not fit into memory.
        The computation involves reshaping and broadcasting the feature arrays to calculate the pairwise differences,
        followed by the norm of these differences. The result is a distance matrix raised to the power of `l`.

        Parameters
        ----------
        l : int or float
            The order of the norm to use for calculating distances. Can be any valid value accepted by `numpy.linalg.norm`
            and `dask.array.linalg.norm`.

        Returns
        -------
        numpy.ndarray
            The computed distance matrix, with each element [i, j] representing the distance between features[i] and features[j],
            raised to the power of `l`.

        Note
        ----
        This method requires the Dask library for computation. Ensure that Dask is installed and imported. The method assumes
        that `self.features` is initialized and contains the features as a numpy array.
        """
        return np.exp(-distance_matrix/scale)

    
    def training_and_exporting_resutls(self,inputs, # "Layer 0", inputs
                                       upsample = True, # Layer 1, classifier (always trained)
                                       train_embedding = False, # Layer 2, embedder (optional)
                                       output_covariance = False # Layer 3, covariance (optional)
                                       ):
        """
        Train a classifier and optionally an embedding model, then export the results.

        This method trains a classifier to distinguish between classes based on a given feature matrix and labels. It
        handles class imbalance by optionally upsampling the underrepresented classes. The method also supports training
        a separate embedding model for feature representation in a lower-dimensional space. It exports the classification
        scores, the embeddings (if applicable), and optionally the covariance of the predictions.

        Parameters
        ----------
        inputs : pandas.DataFrame
            The input data containing features, labels, and metadata for both labelled and unlabelled samples.
        upsample : bool, optional
            Whether to upsample the data to address class imbalance, by default True.
        train_embedding : bool, optional
            Whether to train an embedding model to represent features in a lower-dimensional space, by default False.
        output_covariance : bool, optional
            Whether to output the covariance of the classifier's predictions, by default False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the classification scores, embeddings (if trained, otherwise filled with np.nan), and variance of predictions
            (if applicable, otherwise filled with np.nan), indexed similarly to the input DataFrame.

        Raises
        ------
        ValueError
            If the labelled data contains only one class or if there are too few examples of any class.

        Notes
        -----
        - The classifier is trained using a custom scale and noise grid for hyperparameter optimization.
        - The embedding model, if trained, is optimized using a triplet loss function for dimensional reduction.
        - The method ensures that the training and validation sets include examples from all classes.
        - The method optionally calculates the covariance of the predictions based on the trained classifier.
        
        The function internally defines `f_upsample` to handle data upsampling and utilizes TensorFlow callbacks for 
        early stopping and learning rate reduction on plateau during training.
        """
        # The following parameters might be changed by the users in the future
        validation_fraction = 0.2
        scale_grid = 2**np.linspace(-2,13,16)
        noise_grid = [1e-8]
        cb = [tf.keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,cooldown=5)]
        
        # Function(s) definition
        def f_upsample(x,y,random_state=42):
            """
            Perform upsampling on the dataset to balance class distribution.

            This function addresses class imbalance by upsampling the under-represented classes in the dataset. For each class
            that has fewer samples than the most populous class, it resamples (with replacement) its instances until all classes
            have the same number of samples. The dataset is then shuffled to ensure randomness in the ordering of samples.

            Parameters
            ----------
            x : numpy.ndarray
                The feature array with shape (n_samples, n_features), where 'n_samples' is the number of samples and
                'n_features' is the number of features.
            y : numpy.ndarray
                The label array with shape (n_samples, 1), where each entry is the class label for the corresponding sample in 'x'.
            random_state : int, optional
                A seed value to ensure reproducibility of the random operations, by default 42.

            Returns
            -------
            tuple
                A tuple containing three elements:
                - The upsampled feature array.
                - The upsampled label array.
                - The indices of the samples in the upsampled dataset after shuffling.

            Example
            -------
            >>> x_upsampled, y_upsampled, idx = f_upsample(x, y)
            >>> print(f'Upsampled dataset size: {x_upsampled.shape[0]}')

            Note
            ----
            The function uses 'np.random.seed' to fix the random state for reproducibility. It employs 'np.unique' to find the
            class distribution, 'sklearn.utils.resample' for upsampling, and 'np.random.shuffle' to shuffle the upsampled dataset.
            Ensure that 'x' and 'y' are numpy arrays and that 'sklearn' is installed in your environment.
            """
            #fix seed
            np.random.seed(random_state)

            # uniuqe classes w/ counts
            y_counts = np.unique(y,return_counts=True)
            # index of the most present class 
            n_class_most = np.max(y_counts[1])
            
            x_temp = x.copy()
            y_temp = y.copy()
            
            for _class,_n_class in zip(*y_counts):
                if _n_class != n_class_most:
                    x_2 = resample(x[y.flatten()==_class,...],n_samples=n_class_most-_n_class)
                    y_2 = np.ones((x_2.shape[0],1),dtype=y.dtype)*_class
                
                    x_temp = np.concatenate((x_temp,x_2),axis=0)
                    y_temp = np.concatenate((y_temp,y_2),axis=0)
            
            idx = np.arange(len(y_temp))
            np.random.shuffle(idx)

            return x_temp[idx,...],y_temp[idx,...],idx

        # separate labelled data from unlabelled
        labelled_data = inputs.query('labelled.astype("str") == "1"').astype(int)
        unlabelled_data = inputs.query('labelled.astype("str") == "0"')

        # define a new column reducing the labels to good (==1) or bad (==0)
        labelled_data['label_binary'] = labelled_data['label']>0
        
        # Training - Validation Split:
        # Rise an error asking the user to select some examples of both classes (bad and good)
        if np.unique(labelled_data['label_binary']).size == 1:
            if np.unique(labelled_data['label_binary']) == 0:
                raise ValueError(f'Labelled data contains only bad curves, please select some good examples')
            else:
                raise ValueError(f'Labelled data contains only good curves, please select some bad examples')

        # Ensure that at least 4 curves were selected (2 good and 2 bad) for training-validation split
        temp = np.unique(labelled_data['label_binary'], return_counts = True)[1]
        if (temp < np.array([2,2])).any():
            if (temp[0] >= 2) & (temp[1] < 2):
                raise ValueError(f'Too few good example(s). Please select at least 2 good curves.')
            elif (temp[0] < 2) & (temp[1] >= 2):
                raise ValueError(f'Too few bad example(s). Please select at least 2 bad curves.')
            else:
                raise ValueError(f'Too few example(s). Please select at least 2 bad and 2 good curves.')

        # Ensure that training and validation datasets contain at least one datapoint of each class (bad/good)
        labelled_temp_0 = labelled_data.query('label.astype("str") == "0"').iloc[:2]
        labelled_temp_1 = labelled_data.query('label.astype("str") == "1"').iloc[:2]
        data_train = pd.concat((labelled_temp_0.iloc[:1],
                                     labelled_temp_1.iloc[:1]),axis=0)
        data_valid = pd.concat((labelled_temp_0.iloc[1:],
                                     labelled_temp_1.iloc[1:]),axis=0)
        labelled_data_temp = labelled_data.drop(pd.concat((data_train,data_valid),axis=0).index)

        # Ensure labelled_data_temp still contains data to be splitted
        if labelled_data_temp.shape[0] > 1:
            try:
                data_train_temp, data_valid_temp = train_test_split(labelled_data_temp, test_size = validation_fraction, random_state = 42, shuffle = True, stratify = labelled_data_temp.label_binary.to_numpy().astype(int))
            except:
                data_train_temp, data_valid_temp = train_test_split(labelled_data_temp, test_size = validation_fraction, random_state = 42, shuffle = True)
            data_train = pd.concat((data_train,
                                    data_train_temp))
            data_valid = pd.concat((data_valid,
                                    data_valid_temp))
        elif labelled_data_temp.size == 1:
                data_train = pd.concat((data_train,
                                        labelled_data_temp))
        
        # define indexes
        idx_train = data_train.idx.to_numpy()
        idx_valid = data_valid.idx.to_numpy()

        # upsampling of training and validation datasets to equalise the minority class (nr bad == nr good)
        if upsample:
            idx_train,_,_=f_upsample(idx_train.reshape(-1,1),data_train.loc[:,'label_binary'].to_numpy().astype(int).reshape(-1,1),random_state=42)
            idx_valid,_,_=f_upsample(idx_valid.reshape(-1,1),data_valid.loc[:,'label_binary'].to_numpy().astype(int).reshape(-1,1),random_state=42)
            idx_train = idx_train.flatten()
            idx_valid = idx_valid.flatten()

            # assert that nr bad curves == nr good curves (correctly upsampled)
            assert((np.unique(data_train.reset_index().set_index('idx').loc[idx_train,'label_binary'].to_numpy().astype(int),return_counts=True)[1] == np.unique(data_train.reset_index().set_index('idx').loc[idx_train,'label_binary'].to_numpy().astype(int),return_counts=True)[1].max()).all())
            assert((np.unique(data_valid.reset_index().set_index('idx').loc[idx_valid,'label_binary'].to_numpy().astype(int),return_counts=True)[1] == np.unique(data_valid.reset_index().set_index('idx').loc[idx_valid,'label_binary'].to_numpy().astype(int),return_counts=True)[1].max()).all())


        # distance matrices and binary labels extraction (classification)
        distance_matrix_train_train = self.distance_matrix[idx_train,:][:,idx_train]
        distance_matrix_valid_train = self.distance_matrix[idx_valid,:][:,idx_train]
        y_classifier_train = data_train.reset_index().set_index('idx').loc[idx_train,'label_binary'].to_numpy().astype(int)
        y_classifier_valid = data_valid.reset_index().set_index('idx').loc[idx_valid,'label_binary'].to_numpy().astype(int)

        #hyperparameter optimization
        opt_validation_score = -np.inf
        for scale in scale_grid:
            # define kernel matrices using `scale`
            kernel_matrix_train_train = self.calculate_kernel_matrix(distance_matrix_train_train,scale)
            kernel_matrix_valid_train = self.calculate_kernel_matrix(distance_matrix_valid_train,scale)
            for noise in noise_grid:
                # fit classifier
                self.classifier.fit(kernel_matrix_train_train + noise * np.eye(kernel_matrix_train_train.shape[0]) ,y_classifier_train.flatten())
                # validate the trained model on validation set
                validation_score = self.classifier.score(kernel_matrix_valid_train,y_classifier_valid.flatten())
                # if the new validation_score is better than the previous optimum, export new optimal values 
                if validation_score > opt_validation_score:
                    opt_validation_score = validation_score
                    self.optimal_hyperparameters = {'scale':scale,
                                                    'noise':noise}
                    opt_classifier = self.classifier
        self.classifier = opt_classifier

        # apply the classifier to the whole dataset (labelled + unlabelled) to obtain the scores
        distance_matrix_all_train = self.distance_matrix[:,idx_train]
        kernel_matrix_all_train = self.calculate_kernel_matrix(distance_matrix_all_train, self.optimal_hyperparameters['scale'])
        self.score = self.classifier.predict_proba(kernel_matrix_all_train)[:,1:]


        # Embedding training (if necessary)
        if train_embedding:
            # define new subset of labelled data selecting only good curves
            labelled_data_embedding = labelled_data.query('label_binary == 1')

            # Ensure that at least 2 curves were selected per each Pathway for training-validation split
            temp = np.unique(labelled_data_embedding['label'], return_counts = True)
            if (temp[1] < 2).any():
                # this triggers only a warning since it is still possible to train the embedder (it needs only at least 2 examples of 2 Pathways)
                if (temp[1] >= 2).sum() >= 2:
                    # one or more Pathways lack at least 2 examples. At least 2 examples for each of 2 Pathways were selected (can continue)
                    print('')
                    print('###')
                    print('Warning: please try to select 2 examples per each Pathway.')
                    print(f'         add new examples for Pathway(s) {list(temp[0][temp[1] < 2])}')
                    print('###')
                    print('')
                elif (temp[1].size > 1):
                    # More than one Pathway selected but lacking enough examples
                    raise ValueError(f'Too few example(s). Please select at least 2 examples of at least 2 Pathways. Please add new examples for Pathway(s) {list(temp[0][temp[1] < 2])}')
            if (temp[1].size < 2):
                raise ValueError(f'Too few example(s). Please select at least 2 examples of at least 2 Pathways.')

            # Ensure that training and validation datasets contain at least one datapoint of each class (pathways)
            labelled_data_embedding_temp = []
            for pathway in np.unique(labelled_data_embedding['label']).astype("str"):
                labelled_data_embedding_temp.append(labelled_data_embedding.query(f'label.astype("str") == "{pathway}"').iloc[:2])
            data_train_embedding = pd.concat((labelled_data_embedding_temp),axis=0).iloc[::2]
            data_valid_embedding = pd.concat((labelled_data_embedding_temp),axis=0).iloc[1::2]
            labelled_data_embedding_temp = labelled_data_embedding.drop(pd.concat((data_train_embedding,data_valid_embedding),axis=0).index)

            # Ensure labelled_data_temp still contains data to be splitted
            if labelled_data_embedding_temp.shape[0] > 1:
                try:
                    data_train_embedding_temp, data_valid_embedding_temp = train_test_split(labelled_data_embedding_temp, test_size = validation_fraction, random_state = 42, shuffle = True, stratify = labelled_data_embedding_temp.label.to_numpy().astype(int))
                except:
                    data_train_embedding_temp, data_valid_embedding_temp = train_test_split(labelled_data_embedding_temp, test_size = validation_fraction, random_state = 42, shuffle = True)
                data_train_embedding = pd.concat((data_train_embedding,
                                                  data_train_embedding_temp))
                data_valid_embedding = pd.concat((data_valid_embedding,
                                                  data_valid_embedding_temp))
            elif labelled_data_embedding_temp.size == 1:
                    data_train_embedding = pd.concat((data_train_embedding,
                                                      labelled_data_embedding_temp))

            # define indexes (embedding)
            idx_train_embedding = data_train_embedding.idx.to_numpy()
            idx_valid_embedding = data_valid_embedding.idx.to_numpy()

            # upsampling of training and validation datasets to equalise the minority class (nr class_{z} == nr most present pathway)
            if upsample:
                idx_train_embedding,_,_=f_upsample(idx_train_embedding.reshape(-1,1),data_train_embedding.loc[:,'label'].to_numpy().astype(int).reshape(-1,1),random_state=42)
                idx_valid_embedding,_,_=f_upsample(idx_valid_embedding.reshape(-1,1),data_valid_embedding.loc[:,'label'].to_numpy().astype(int).reshape(-1,1),random_state=42)
                idx_train_embedding = idx_train_embedding.flatten()
                idx_valid_embedding = idx_valid_embedding.flatten()

                # assert that each class is represented in equal numbers (upsampled)
                assert((np.unique(data_train_embedding.reset_index().set_index('idx').loc[idx_train_embedding,'label'].to_numpy().astype(int),return_counts=True)[1] == np.unique(data_train_embedding.reset_index().set_index('idx').loc[idx_train_embedding,'label'].to_numpy().astype(int),return_counts=True)[1].max()).all())
                assert((np.unique(data_valid_embedding.reset_index().set_index('idx').loc[idx_valid_embedding,'label'].to_numpy().astype(int),return_counts=True)[1] == np.unique(data_valid_embedding.reset_index().set_index('idx').loc[idx_valid_embedding,'label'].to_numpy().astype(int),return_counts=True)[1].max()).all())


            # distance matrices and binary labels extraction (embedding)
            distance_matrix_embedding_train_train = self.distance_matrix[idx_train_embedding,:][:,idx_train_embedding]
            distance_matrix_embedding_valid_train = self.distance_matrix[idx_valid_embedding,:][:,idx_train_embedding]
            y_classifier_train_embedding = data_train_embedding.reset_index().set_index('idx').loc[idx_train_embedding,'label'].to_numpy().astype(int)
            y_classifier_valid_embedding = data_valid_embedding.reset_index().set_index('idx').loc[idx_valid_embedding,'label'].to_numpy().astype(int)

            # kernel matrices (embedding). Calculated using the optimal hyperparameters optained during the training of the classifier
            kernel_matrix_embedding_train_train = self.calculate_kernel_matrix(distance_matrix_embedding_train_train, self.optimal_hyperparameters['scale'])
            kernel_matrix_embedding_valid_train = self.calculate_kernel_matrix(distance_matrix_embedding_valid_train, self.optimal_hyperparameters['scale'])


            # reinitialize embedding model at each iteration
            self.embedder = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
            self.embedder.compile(loss=TripletHardLoss())
            
            # training embedder
            self.embedder.fit(kernel_matrix_embedding_train_train,y_classifier_train_embedding,
                              epochs=10000,
                              validation_data=(kernel_matrix_embedding_valid_train,y_classifier_valid_embedding),
                              callbacks=cb,verbose=0)

            # apply the embedder to the whole dataset (labelled + unlabelled) to obtain the embedding scores
            distance_matrix_embedding_all_train = self.distance_matrix[:,idx_train_embedding]
            kernel_matrix_embedding = self.calculate_kernel_matrix(distance_matrix_embedding_all_train, self.optimal_hyperparameters['scale'])
            
            self.embedding = self.embedder.predict(kernel_matrix_embedding)
        else:
            # if embedder is not necessary
            self.embedder = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
            self.embedder.predict(np.array([[1]]))
            self.embedding = np.nan * np.ones((self.score.shape[0],self.embedder.layers[-1].output_shape[-1]))

        if output_covariance:
             # kernel matrices
            kernel_matrix_train_train = self.calculate_kernel_matrix(distance_matrix_train_train, self.optimal_hyperparameters['scale'])
            kernel_matrix_all_train = self.calculate_kernel_matrix(distance_matrix_all_train, self.optimal_hyperparameters['scale'])
            
            # cholesky decomposition to invert the training kernel marix 
            c, low = cho_factor(kernel_matrix_train_train + self.optimal_hyperparameters['noise'] * np.eye(kernel_matrix_train_train.shape[0]))
            kernel_matrix_inv_train_train = cho_solve((c, low), np.eye(kernel_matrix_train_train.shape[0]))
            
            # calculate the posterior covariance
            self.covariance = (1-np.diag(kernel_matrix_all_train@kernel_matrix_inv_train_train@kernel_matrix_all_train.T)).reshape(-1,1)
        else:
            # if covariance is not necessary
            self.covariance = np.nan * np.ones((self.score.shape[0],1))

        # output of Layer 1 (classifier)
        classifier_output = pd.DataFrame(self.score.copy(),
                                         index = inputs.index,
                                         columns = pd.MultiIndex.from_tuples([("1_classifier","score")], names=['Layer', 'Column']))
        
        # output of Layer 2 (embedder)
        embedding_output = pd.DataFrame(self.embedding.copy(),
                                        index = inputs.index,
                                        columns = pd.MultiIndex.from_tuples([("2_embedding",f'emb_{i}') for i in range(self.embedding.shape[1])], names=['Layer', 'Column']))
        
        # output of Layer 3 (covariance)
        covariance_output = pd.DataFrame(self.covariance.copy(),
                                         index = inputs.index,
                                         columns = pd.MultiIndex.from_tuples([("3_covarince",'covariance')], names=['Layer', 'Column']))

        # merge outpurs
        outputs = pd.concat((classifier_output,
                             embedding_output,
                             covariance_output),axis=1)
        
        return outputs




if __name__ == '__main__':
    #load images and total labels exported with Export_AFM_data_demo
    (images,total_labels) = load_images_labels_from_npz()

    # print(images[0])
    # [[  0   0   0 ...   0   0   0]
    #  [  0   0   0 ...   0   0   0]
    #  [  0   0   0 ...   0   0   0]
    #               ...
    #  [  0   0   0 ...   0   0   0]
    #  [  0   0   0 ...   0   0 255]
    #  [255 255 255 ... 255 255 255]]

    # print(total_labels)
    # np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3]])


    #select only few curves to simulate a real situation where only some curves were labelled by the user
    selected_indexes = np.array([0,1,2,# [ 0 -  4] Bad curves selected 
                                 5,6,  # [ 5 -  9] Pathway 1 curves selected 
                                 10,11,# [10 - 14] Pathway 2 curves selected (nothing selected) 
                                 #       [15 - 19] Pathway 3 curves selected (nothing selected) 
                                 ])
    
    
    #create the inputs for the docker container
    docker_inputs = get_inputs_4_docker_cont(total_labels,selected_indexes)

    # print(docker_inputs)
    #        idx labelled label
    # AFM_00   0        1     0
    # AFM_01   1        1     0
    # AFM_02   2        1     0
    # AFM_03   3        0  <NA>
    # AFM_04   4        0  <NA>
    # AFM_05   5        1     1
    # AFM_06   6        1     1
    # AFM_07   7        1     1
    # AFM_08   8        0  <NA>
    # AFM_09   9        0  <NA>
    # AFM_10  10        1     2
    # AFM_11  11        1     2
    # AFM_12  12        0  <NA>
    # AFM_13  13        0  <NA>
    # AFM_14  14        0  <NA>
    # AFM_15  15        0  <NA>
    # AFM_16  16        0  <NA>
    # AFM_17  17        0  <NA>
    # AFM_18  18        0  <NA>
    # AFM_19  19        0  <NA>

    # Instantiating FusionLearning class, which simulate what a docker container should do:
    #   1) Use the curves (images) to extract features from DenseNet (convolution neural net)
    #   2) Use the features to calculate the (euclidian) distance matrix between ALL datapoints in the set
    #   3) Store both arrays in self.features and self.distance_matrix
    #   4) Train all models and exporting the outputs (by calling the method "training_and_exporting_resutls", see below)
    print('Instantiating')
    fusion_learning = FusionLearning(images)
    print('')

    # print(fusion_learning.features[:2])
    # [[1.0577803e-04 3.4208356e-03 1.1759967e-03 ... 9.1565900e-02
    #     5.5742610e-01 1.7885832e-01]
    # [1.0214470e-04 4.5245951e-03 1.0156455e-03 ... 4.2897236e-01
    #     7.3882961e-01 7.0256487e-02]]
    
    # print(fusion_learning.distance_matrix[:2,:][:,:2])
    # [[ 0.        55.9319168]
    #  [55.9319168  0.       ]]

    # Training all models
    # This may includ all or some Fusion Learning "layers"
    # Layer 1 (necessary): Classifier, which output a score calculated on each curve. This should be used to reorder the data (labelled and unlabelled)
    # Layer 2 (optional):  Neural network, which output an embedding score (--> 2D array) on each curve. This should be used in the GUI to project the data in the embedding.
    # Layer 3 (optional):  Covariance, which output a covariance score on each curve describing the affinity with the training set. This should also be used in the GUI as an additional dimension (with the embedding of Layer 2) to visualise the data.
    print('Iteration 0')
    docker_outputs = fusion_learning.training_and_exporting_resutls(docker_inputs, # "Layer 0", inputs
                                                                    upsample = True, # Layer 1, classifier (always trained)
                                                                    train_embedding = False, # Layer 2, embedder (optional)
                                                                    output_covariance = False # Layer 3, covariance (optional)
                                                                    )
    # Export Data
    docker_inputs.columns = pd.MultiIndex.from_tuples([("0_inputs",i) for i in docker_inputs.columns], names=['Layer', 'Column'])
    df = pd.concat((docker_inputs,docker_outputs),axis=1)
    df.to_csv('df_0000.csv')

    # print(df.head())
    # Layer    0_inputs                 ... 2_embedding         3_covarince
    # Column        idx labelled label  ...       emb_0 emb_1    covariance
    # AFM_0000        0        1     0  ...         NaN   NaN  3.714003e-09
    # AFM_0001        1        1     0  ...         NaN   NaN  1.960893e-01
    # AFM_0002        2        0  <NA>  ...         NaN   NaN  9.889361e-01
    # AFM_0003        3        0  <NA>  ...         NaN   NaN  1.319419e-01
    # AFM_0004        4        0  <NA>  ...         NaN   NaN  8.343802e-01
    # [5 rows x 7 columns]

    # Additional Iterations
    selected_indexes_new = selected_indexes.copy()
    n_new_curves_per_iteration = 7
    print('')
    for i in range(1,3):#range(total_labels.size):
        print(f'Iteration {i}')

        # label the unlabelled datapoint(s) with the highest score and append it(them) to the labelled dataset
        selected_indexes_new = np.concatenate((selected_indexes_new,df.droplevel(level=0, axis=1).query('labelled.astype("str") == "0"').sort_values("score",ascending=False).iloc[:n_new_curves_per_iteration].idx.to_numpy().astype(int).flatten()))
        # recalculate the inputs (with the new datapoints)
        docker_inputs_new = get_inputs_4_docker_cont(total_labels,selected_indexes_new)
        # retrain models 
        docker_outputs_new = fusion_learning.training_and_exporting_resutls(docker_inputs_new, # "Layer 0", inputs
                                                                            upsample = True, # Layer 1, classifier (always trained)
                                                                            train_embedding = True, # Layer 2, embedder (optional)
                                                                            output_covariance = True # Layer 3, covariance (optional)
                                                                            )

        # export iteration
        docker_inputs_new.columns = pd.MultiIndex.from_tuples([("0_inputs",i) for i in docker_inputs_new.columns], names=['Layer', 'Column'])
        df = pd.concat((docker_inputs_new,docker_outputs_new),axis=1)
        df = df.loc[df.droplevel(level=0, axis=1).sort_values('score').index,:]
        df.to_csv(f'df_{i:04d}.csv')
        
        print('')

    