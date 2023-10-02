#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:22:13 2023

@author: vdoffini
"""

import argparse 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d","--dataset", help="dataset (case study) used (experiment / simulation)",
                    action="store",default='experiment')
parser.add_argument("-n","--n_iter_new_curves", type=int, help="How many traces visualize at each new iteration",
                    action="store",default=4)
parser.add_argument("--data_emb",nargs='+', type=int, help="At which iteration should perform embedding? It has to be 3 values!",
                    action="store",default=[20,500,1000])
parser.add_argument("-l", type=float, help="What norm use for the distance matrix",
                    action="store",default=2)
parser.add_argument("--savedata", type=bool, help="Save all data (general and per each iteration in) in folder ./ExportedData",
                    action="store",default=False)
parser.add_argument("--savefig", type=bool, help="Save Figure in folder ./Figures",
                    action="store",default=False)
parser.add_argument("--upsample", type=bool, help="Perform upsampling on data?",
                    action="store",default=True)
args = parser.parse_args()

#%% Load Libraries
import tensorflow as tf #tf.__version__ == 2.6.0
from tensorflow_addons.losses import TripletHardLoss #tfa.__version__ == 0.14.0
import numpy as np #np.__version__ == 1.19.2
import math
import matplotlib.pyplot as plt #matplotlib.__version__ == 3.3.1
# plt.rcParams["font.family"] = "Times"#https://stackoverflow.com/questions/40734672/how-to-set-the-label-fonts-as-time-new-roman-by-drawparallels-in-python/40734893
plt.rcParams['text.usetex'] = True 
plt.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
from matplotlib.cm import cool,viridis
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.path as mplPath
from sklearn.cluster import KMeans #sklearn.__version__ == 0.23.1
kmeans = KMeans(n_clusters=2,random_state=42,n_jobs=1)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import dask.array as da #dask.__version__ == 2021.07.0
from dask.diagnostics import ProgressBar

cmap_bad_good_simulated = LinearSegmentedColormap.from_list("", [np.array([0.8834,0,0.4756,1]),np.array([0,0.5625,0.2128,1])])
cmap_bad_1_2_3_simulated = LinearSegmentedColormap.from_list("", [np.array([0.8834,0,0.4756,1]),np.array([0.5611,0.7765,0.5875,1]),np.array([0.5150,0.8113,0.9383,1]),np.array([1,0.9579,0.6038,1])])
cmap_bad_1_2_3_exp = LinearSegmentedColormap.from_list("", [np.array([1,0,0,1]),viridis(0),viridis(0.5),viridis(0.999)])

#%% Parameters

# save plots?
bool_savefigs = args.savefig

# export data?
bool_exportdata = args.savedata

# data upsample?
upsampl = args.upsample

# data set to use
dataset = args.dataset.lower()
if dataset == 'experiment':
    file_name_dataset = 'Experiment.npz'
elif dataset == 'simulation':
    file_name_dataset = 'Simulation_train.npz'
else:
    raise(ValueError('dataset should be "experiment" or "simulation"'))

# distance marix norm (l=2 --> norm 2)
l=args.l#2

# new curves at each iteration
n_iter_new_curves = args.n_iter_new_curves#10

# datapoints where to train the embedding space neural network. len(data_emb) == 3
data_emb = np.sort(args.data_emb)
if len(data_emb) != 3:
    raise(ValueError('data_emb should contain 3 values'))

# kernel scale span
kernel_scales = 2**np.linspace(-2,13,16)

#%% Functions
def upsample(x,y,random_state=42):
    np.random.seed(random_state)
    y_counts = np.unique(y,return_counts=True)[1]
    class_larger = np.argmax(y_counts)
    class_smaller = np.argmin(y_counts)

    x_2 = resample(x[y.flatten()==class_smaller,...],n_samples=y_counts[class_larger]-y_counts[class_smaller])
    y_2 = np.ones((x_2.shape[0],1),dtype=y.dtype)*class_smaller

    idx = np.arange(len(y)+len(y_2))
    np.random.shuffle(idx)

    x_temp = np.concatenate((x,x_2),axis=0)
    y_temp = np.concatenate((y,y_2),axis=0)

    return x_temp[idx,...],y_temp[idx,...],idx
    
def upsample_emb(x,y,random_state=42):
    np.random.seed(random_state)
    y_counts = np.unique(y,return_counts=True)
    idx_class_larger = np.argmax(y_counts[1])
    
    x_temp = x.copy()
    y_temp = y.copy()
    
    for class_smaller,n_class_smaller in zip(y_counts[0][np.argsort(y_counts[1])][:-1],y_counts[1][np.argsort(y_counts[1])][:-1]):

        x_2 = resample(x[y.flatten()==class_smaller,...],n_samples=y_counts[1][idx_class_larger]-n_class_smaller)
        y_2 = np.ones((x_2.shape[0],1),dtype=y.dtype)*class_smaller
    
        x_temp = np.concatenate((x_temp,x_2),axis=0)
        y_temp = np.concatenate((y_temp,y_2),axis=0)
    
    idx = np.arange(len(y_temp))
    np.random.shuffle(idx)

    return x_temp[idx,...],y_temp[idx,...],idx

def import_data(file_name_dataset = 'Experiment.npz'):
    with np.load('./Data/'+file_name_dataset,allow_pickle = True) as f:
        x_raw = f['x_raw']
        y_raw = f['y_raw']
        x_1D_processed = f['x_1D_processed']
        x_2D_processed = f['x_2D_processed'][...,np.newaxis].astype(np.uint8)
        
    idx_main = np.arange(len(x_raw))
    if False:
        # print('shuffle = True\n')
        np.random.shuffle(idx_main)
        y_raw = y_raw[idx_main,...]
        x_raw = x_raw[idx_main,...]
        x_1D_processed = x_1D_processed[idx_main,...]
        x_2D_processed = x_2D_processed[idx_main,...]
    else:
        # print('shuffle = False\n')
        1
    
    if file_name_dataset == 'Experiment.npz':
        # print('Attention: y_raw = y_raw+1; y_raw[y_raw[:,0]==4]=0\n')
        y_raw = y_raw+1
        y_raw[y_raw[:,0]==4]=0
    
    x = tf.keras.applications.densenet.preprocess_input(tf.image.resize(255-np.repeat(x_2D_processed,3,axis=-1).astype(np.float32),(256,256),'nearest'))
    y = (y_raw>0).astype(int)
    return x,y,x_raw,y_raw,x_1D_processed,x_2D_processed

def distance(x1,x2,l):
    ProgressBar().register()
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

def line_eq(x,centroids):
    #https://math.stackexchange.com/questions/771761/equation-of-a-line-that-passes-halfway-between-two-points-in-other-words-divid
    a=centroids[0,0]
    b=centroids[0,1]
    c=centroids[1,0]
    d=centroids[1,1]
    return (a**2+b**2-c**2-d**2-2*np.array(x)*(a-c))/(2*(b-d))

def angle(v1, v2):
    #https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    def dotproduct(v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))
    def length(v):
        return math.sqrt(dotproduct(v, v))
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def R_matrix(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
#%% Load Data
x,y,x_raw,y_raw,x_1D_processed,x_2D_processed = import_data(file_name_dataset)
mask_train = y_raw.flatten()!=3

# shuffle
idx_train = np.arange(mask_train.sum())
np.random.seed(0)
np.random.shuffle(idx_train)
    
if file_name_dataset == 'Simulation_train.npz':
    cmap_bad_good = cmap_bad_good_simulated
    cmap_bad_1_2_3 = cmap_bad_1_2_3_simulated
    
    x=x.numpy()[mask_train,...][idx_train,...]
    y=y[mask_train,...][idx_train,...]
    x_raw=x_raw[mask_train,...][idx_train,...]
    x_1D_processed=x_1D_processed[mask_train,...][idx_train,...]
    x_2D_processed=x_2D_processed[mask_train,...][idx_train,...]
    y_raw=y_raw[mask_train,...][idx_train,...]
    
    file_name_dataset = 'Simulation_test.npz'
    x_test,y_test,x_raw_test,y_raw_test,x_1D_processed_test,x_2D_processed_test = import_data(file_name_dataset)

else:
    cmap_bad_good = cool
    cmap_bad_1_2_3 = cmap_bad_1_2_3_exp
    
    x,y,x_raw,y_raw,x_1D_processed,x_2D_processed = import_data(file_name_dataset)
        
    x_test=np.concatenate((x.numpy()[mask_train,...][idx_train[1100:],...],x.numpy()[np.logical_not(mask_train),...]),axis=0)
    y_test=np.concatenate((y[mask_train,...][idx_train[1100:],...],y[np.logical_not(mask_train),...]),axis=0)
    x_raw_test=np.concatenate((x_raw[mask_train,...][idx_train[1100:],...],x_raw[np.logical_not(mask_train),...]),axis=0)
    x_1D_processed_test=np.concatenate((x_1D_processed[mask_train,...][idx_train[1100:],...],x_1D_processed[np.logical_not(mask_train),...]),axis=0)
    x_2D_processed_test=np.concatenate((x_2D_processed[mask_train,...][idx_train[1100:],...],x_2D_processed[np.logical_not(mask_train),...]),axis=0)
    y_raw_test=np.concatenate((y_raw[mask_train,...][idx_train[1100:],...],y_raw[np.logical_not(mask_train),...]),axis=0)
    
    x=x.numpy()[mask_train,...][idx_train[:1100],...]
    y=y[mask_train,...][idx_train[:1100],...]
    x_raw=x_raw[mask_train,...][idx_train[:1100],...]
    x_1D_processed=x_1D_processed[mask_train,...][idx_train[:1100],...]
    x_2D_processed=x_2D_processed[mask_train,...][idx_train[:1100],...]
    y_raw=y_raw[mask_train,...][idx_train[:1100],...]
    
y=y.flatten()[:,np.newaxis]
y_test=y_test.flatten()[:,np.newaxis]
y_raw=y_raw.flatten()[:,np.newaxis]
y_raw_test=y_raw_test.flatten()[:,np.newaxis]


#%% Featurization (image --> vector)
model_vanilla = tf.keras.applications.DenseNet121()
model_vanilla_emb = tf.keras.models.Model(inputs=model_vanilla.input,outputs=model_vanilla.layers[-2].output)
x_emb_train = model_vanilla_emb.predict(tf.keras.applications.densenet.preprocess_input(tf.image.resize(255-np.repeat(x_2D_processed,3,axis=-1).astype(np.float32),(224,224),'nearest')))
x_emb_test = model_vanilla_emb.predict(tf.keras.applications.densenet.preprocess_input(tf.image.resize(255-np.repeat(x_2D_processed_test,3,axis=-1).astype(np.float32),(224,224),'nearest')))

#%% Distance matrix (norm-l)
distance_trtot_trtot = distance(x_emb_train,x_emb_train,l)
distance_te_trtot = distance(x_emb_test,x_emb_train,l)
distance_te_te = distance(x_emb_test,x_emb_test,l)


#%% Iterative procedure

# fix tensorflow seed
tf.random.set_seed(42)

# initialize fraction of good curves found
frac_data = [0]
result_frac_positive=[0]

#data
data = np.arange(1,np.ceil(len(x_emb_train)/n_iter_new_curves)+1).astype(int)*n_iter_new_curves

# callbacks for embedding space neural network
cb = [tf.keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True)
        ,tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,cooldown=5)]

# output figure
fig_out = plt.figure(figsize=(9.5,9.5*2480/2391))
gs = fig_out.add_gridspec(4,6)#, hspace=0.1, wspace=0.1)
gs.set_height_ratios([1,2,0.,2])
ax_layer1 = fig_out.add_subplot(gs[0, :])# binary classifier
ax_layer1.set_xlim(0,data.max())
ax_layer1.set_ylim(0,100)
ax_layer1.plot([0,data.max()],[0,100],'k--')
ax_layer2_0 = fig_out.add_subplot(gs[1, 0:2])# embedding @data_emb[0]
ax_layer2_1 = fig_out.add_subplot(gs[1, 2:4])# embedding @data_emb[1]
ax_layer2_2 = fig_out.add_subplot(gs[1, 4:6])# embedding @data_emb[2]
ax_legend = fig_out.add_subplot(gs[2, 0:6])
ax_layer3_0 = fig_out.add_subplot(gs[3, 0:3])# covariance @data_cov, p_classifier>0.5
ax_layer3_1 = fig_out.add_subplot(gs[3, 3:6])# covariance @data_cov, p_classifier<0.5

# output figure legend
ax_legend.scatter([0],[0],c=cmap_bad_1_2_3(np.array([0/3])),label='Bad',marker='.', edgecolor='black', s=60*2,linewidth=0.1)
ax_legend.scatter([0],[0],c=cmap_bad_1_2_3(np.array([1/3])),label='P1',marker='.', edgecolor='black', s=60*2,linewidth=0.1)
ax_legend.scatter([0],[0],c=cmap_bad_1_2_3(np.array([2/3])),label='P2',marker='.', edgecolor='black', s=60*2,linewidth=0.1)
ax_legend.scatter([0],[0],c=cmap_bad_1_2_3(np.array([3/3])),label='P3',marker='.', edgecolor='black', s=60*2,linewidth=0.1)
ax_legend.set_axis_off()
ax_legend.legend(loc="center", ncol = 4, columnspacing=5, prop={'size': 12})

# initialize export data 
if bool_exportdata:
    import pickle 
    d_general = {}
    d_general['x'] = x
    d_general['y'] = y
    d_general['x_raw'] = x_raw
    d_general['y_raw'] = y_raw
    d_general['x_1D_processed'] = x_1D_processed
    d_general['x_2D_processed'] = x_2D_processed
    d_general['x_emb_train'] = x_emb_train
    d_general['x_test'] = x_test
    d_general['y_test'] = y_test
    d_general['x_raw_test'] = x_raw_test
    d_general['y_raw_test'] = y_raw_test
    d_general['x_1D_processed_test'] = x_1D_processed_test
    d_general['x_emb_test'] = x_emb_test
    d_general['idx_train'] = idx_train #train set shuffling
    d_general['distance_trtot_trtot'] = distance_trtot_trtot
    d_general['distance_te_trtot'] = distance_te_trtot
    d_general['distance_te_te'] = x_emb_test

    d_iter = {}
    

# FUSION Learning core 
for i,i_data in enumerate(data):
    # initialize dictionary for export iterational data
    if bool_exportdata:
        d_iter_temp = {}
    
    # idx_visualized == index of data that we have access (training_tot_training + training_tot_validation)
    # idx_visualized_shuff_class == idx_visualized shuffled for classification
    # idx_visualized_shuff_emb == idx_visualized shuffled for embedding

    # index of data visualized
    if (i>0)&((i_data-data[i-1])>0):
        # append the new visualized curves
        idx_visualized = np.concatenate((idx_visualized,idx_new_data[:int(i_data-data[i-1])]))
    else:
        # initialize
        idx_visualized = np.arange(i_data)
    
    # training / validation split (binary classifier)
    np.random.seed(0)
    idx_visualized_shuff_class = np.random.choice(idx_visualized,size=len(idx_visualized),replace=False)
    nr_data_class_tr = int(len(idx_visualized_shuff_class)*0.8)
    # training index 
    idx_visualized_shuff_class_tr = idx_visualized_shuff_class[:nr_data_class_tr]
    # validation index 
    idx_visualized_shuff_class_va = idx_visualized_shuff_class[nr_data_class_tr:]
    # remaining index (unscreened data)
    idx_visualized_shuff_class_re = np.delete(np.arange(len(x_emb_train)),idx_visualized_shuff_class)
    
    # upsampling (binary classifier)
    if upsampl:
        d_class_tr_tr,y_class_tr_tr,_=upsample(distance_trtot_trtot[idx_visualized_shuff_class_tr,:][:,idx_visualized_shuff_class_tr],y[idx_visualized_shuff_class_tr,:],random_state=42)
        d_class_va_tr,y_class_va_tr,_=upsample(distance_trtot_trtot[idx_visualized_shuff_class_va,:][:,idx_visualized_shuff_class_tr],y[idx_visualized_shuff_class_va,:],random_state=42)
    else:
        d_class_tr_tr = distance_trtot_trtot[idx_visualized_shuff_class_tr,:][:,idx_visualized_shuff_class_tr]
        y_class_tr_tr = y[idx_visualized_shuff_class_tr,:]
        d_class_va_tr = distance_trtot_trtot[idx_visualized_shuff_class_va,:][:,idx_visualized_shuff_class_tr]
        y_class_va_tr = y[idx_visualized_shuff_class_va,:]
        
    
    # hyperparameter optimization (binary classifier)
    score_class_val = []
    logr = LogisticRegression(max_iter=10000,solver='lbfgs',tol=0.0001)
    for k_scale_class in kernel_scales:
        # kernel definition
        k_class_tr_tr = np.exp(-d_class_tr_tr/k_scale_class)
        k_class_va_tr = np.exp(-d_class_va_tr/k_scale_class)
        
        # fit binary classifier
        logr.fit(k_class_tr_tr,y_class_tr_tr.flatten())
        
        # score binary classifier
        score_class_val.append(logr.score(k_class_va_tr,y_class_va_tr))
    
    # recall optimal kernel scale
    k_scale_class_opt = kernel_scales[np.argmax(score_class_val)]
    
    # re-fit binary classifier using the optimal kernel scale
    logr.fit(np.exp(-d_class_tr_tr/k_scale_class_opt),y_class_tr_tr.flatten())

    # if there are still unscreened data, reorder the data accordingly with the binary classifier output
    if not(idx_visualized_shuff_class_re.size == 0):
        prob_pred_re = logr.predict_proba(np.exp(-distance_trtot_trtot[idx_visualized_shuff_class_re,:][:,idx_visualized_shuff_class_tr]/k_scale_class_opt))
        idx_new_data = idx_visualized_shuff_class_re[prob_pred_re[:,0].argsort()]
    
    # predict on test set (binary classifier)
    prob_pred_test = logr.predict_proba(np.exp(-distance_te_trtot[:,:][:,idx_visualized_shuff_class_tr]/k_scale_class_opt))
    y_pred_test = logr.predict(np.exp(-distance_te_trtot[:,:][:,idx_visualized_shuff_class_tr]/k_scale_class_opt))
    
    temp_count = np.histogram(y_raw[idx_visualized],bins=[-0.5,0.5,1.5,2.5,3.5])[0]
    print(f'data   screened: {idx_visualized.size} --> bad curves: {temp_count[0]}; P1: {temp_count[1]}; P2: {temp_count[2]}; P3: {temp_count[3]}')
    temp_count = np.histogram(y_raw[np.delete(np.arange(y_raw.size),idx_visualized)],bins=[-0.5,0.5,1.5,2.5,3.5])[0]
    print(f'data unscreened: {y.size-idx_visualized.size} --> bad curves: {temp_count[0]}; P1: {temp_count[1]}; P2: {temp_count[2]}; P3: {temp_count[3]}')
    temp_count = np.histogram(y_raw_test[y_pred_test.astype(bool),0],bins=[-0.5,0.5,1.5,2.5,3.5])[0]
    print(f'data (test set): {y_raw_test.size} --> bad curves: {temp_count[0]}; P1: {temp_count[1]}; P2: {temp_count[2]}; P3: {temp_count[3]}')
    print('')
    
    # append the new fraction of data visualized and good curves found
    frac_data.append(i_data/y_raw.size)
    result_frac_positive.append(y[idx_visualized,:].sum()/y.sum())
    ax_layer1.plot([i_data-n_iter_new_curves,i_data],np.array([i_data-n_iter_new_curves,i_data])/(y_raw>0).sum()*100,':',color='k',)
    ax_layer1.plot([i_data-n_iter_new_curves,i_data],np.array(result_frac_positive[-2:])*100,color='k')
    plt.pause(0.0001)
    
    # Embedding Space
    # training / validation split (binary classifier)
    idx_visualized_shuff_emb = np.random.choice(idx_visualized[y_raw[idx_visualized].flatten()!=0],size=np.sum(y_raw[idx_visualized].flatten()!=0),replace=False)
    nr_data_emb_tr = int(len(idx_visualized_shuff_emb)*0.8)
    # training index 
    idx_visualized_shuff_emb_tr = idx_visualized_shuff_emb[:nr_data_emb_tr]
    # validation index 
    idx_visualized_shuff_emb_va = idx_visualized_shuff_emb[nr_data_emb_tr:]
    # remaining index 
    idx_visualized_shuff_emb_re = np.delete(np.arange(len(y_raw.flatten())),idx_visualized_shuff_emb)
    idx_visualized_shuff_emb_re = idx_visualized_shuff_emb_re[y_raw.flatten()[idx_visualized_shuff_emb_re]!=0]
    
    # upsampling (embedding space)
    if upsampl:
        d_emb_tr_tr,y_emb_tr_tr,_=upsample_emb(distance_trtot_trtot[idx_visualized_shuff_emb_tr,:][:,idx_visualized_shuff_emb_tr],y_raw[idx_visualized_shuff_emb_tr,:],random_state=42)
        d_emb_va_tr,y_emb_va_tr,_=upsample_emb(distance_trtot_trtot[idx_visualized_shuff_emb_va,:][:,idx_visualized_shuff_emb_tr],y_raw[idx_visualized_shuff_emb_va,:],random_state=42)
    else:
        d_emb_tr_tr = distance_trtot_trtot[idx_visualized_shuff_emb_tr,:][:,idx_visualized_shuff_emb_tr]
        y_emb_tr_tr = y_raw[idx_visualized_shuff_emb_tr,:]
        d_emb_va_tr = distance_trtot_trtot[idx_visualized_shuff_emb_va,:][:,idx_visualized_shuff_emb_tr]
        y_emb_va_tr = y_raw[idx_visualized_shuff_emb_va,:]
    
    #export data
    if bool_exportdata:
        d_iter_temp['data visualized'] = i_data
        d_iter_temp['idx_visualized'] = idx_visualized
        d_iter_temp['fraction of data visualized'] = frac_data[-1]
        d_iter_temp['fraction of good traces visualized'] = result_frac_positive[-1]
        d_iter_temp['optimal kernel scale (classifier)'] = k_scale_class_opt
        d_iter_temp['logistic_regression'] = logr
    
    if i_data in data_emb:
        # select the figure axis for Layer 2
        if i_data==data_emb[0]:
            ax_temp = ax_layer2_0
        elif i_data==data_emb[1]:
            ax_temp = ax_layer2_1
        else:
            ax_temp = ax_layer2_2
        
        
        score_emb_val = []
        # recall optimal kernel scale from Layer 1
        k_scale_emb_opt = k_scale_class_opt
        
        # kernel definition
        k_emb_tr_tr = np.exp(-d_emb_tr_tr/k_scale_emb_opt)
        k_emb_va_tr = np.exp(-d_emb_va_tr/k_scale_emb_opt)
        
        # initialize neural network
        tf.random.set_seed(42)
        model_emb = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
        # model_emb(k_emb_tr_tr)
        # print(model_emb.weights)
        model_emb.compile(loss=TripletHardLoss())
        # fit embedding space
        model_emb.fit(k_emb_tr_tr,y_emb_tr_tr,epochs=10000,validation_data=(k_emb_va_tr,y_emb_va_tr),callbacks=cb,verbose=0)
        
        # evaluate on test set
        y_pred_emb = model_emb.predict(np.exp(-distance_te_trtot[:,idx_visualized_shuff_emb_tr]/k_scale_emb_opt))
        
        print('\n')
        print('Layer 1 (binary classifier) confusion matrix:')
        print(confusion_matrix(y_raw_test,y_pred_test)[:,:2])

        # plot the embedding space (predicted positive accordingly with the binary classifier)
        ax_temp.scatter(y_pred_emb[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]>0.5,0],y_pred_emb[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]>0.5,1],c=cmap_bad_1_2_3(y_raw_test.flatten()[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]>0.5]/3),marker='.', edgecolor='black', s=60,linewidth=0.1)
        ax_temp.scatter(y_pred_emb[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]>0.5,0],y_pred_emb[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]>0.5,1],c=cmap_bad_1_2_3(y_raw_test.flatten()[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]>0.5]/3),marker='.', edgecolor='black', s=60,linewidth=0.1)
        
        ax_temp.axis('square')
        
        #export data
        if bool_exportdata:
            d_iter_temp['optimal kernel scale (embedding)'] = k_scale_emb_opt
            d_iter_temp['embedding neural netowrk'] = f'saved as ./ExportedData/model_emb_{dataset}_{i_data}'
            model_emb.save(f'./ExportedData/model_emb_{dataset}_{i_data}')
            d_iter_temp['embedding components (positive classified)'] = y_pred_emb[y_pred_test.flatten()>0.5,:]
            d_iter_temp['embedding components (negative classified)'] = y_pred_emb[y_pred_test.flatten()<0.5,:]
            
        # if there are more then 2 points in the embedding space, fit kmeans
        if (y_pred_test.flatten()>0.5).sum()>2:
            # fit kmenas on the embedding space (predicted positive accordingly with the binary classifier)
            kmeans_pred = kmeans.fit_predict(y_pred_emb[y_pred_test.flatten()>0.5])#,y_raw_test[y_pred_test.flatten()>0.5]-1)
            
            # extract kmeans center of masses
            kmeans_centers = kmeans.cluster_centers_.copy()
            
            ax_temp_xlim = np.array(ax_temp.get_xlim())
            ax_temp_ylim = np.array(ax_temp.get_ylim())
            
            # define kmeans confusion matrix
            temp_0 = np.unique(y_raw_test[y_pred_test.flatten()>0.5,:][kmeans_pred==0,0],return_counts=True)
            temp = np.zeros((2,4));temp[0,:]=np.arange(4)
            temp[1,temp_0[0]] = temp_0[1]
            temp_0 = temp
            temp_1 = np.unique(y_raw_test[y_pred_test.flatten()>0.5,:][kmeans_pred==1,0],return_counts=True)
            temp = np.zeros((2,4));temp[0,:]=np.arange(4)
            temp[1,temp_1[0]] = temp_1[1]
            temp_1 = temp
            kmeans_confusion_matrix = np.concatenate((temp_0,temp_1),axis=0)[[1,3],:].T
            print('\n')
            print('Layer 2 (embedding, k-means) confusion matrix:')
            print(kmeans_confusion_matrix)

            # manage color scheme of kmeans
            if kmeans_confusion_matrix[1,0]+kmeans_confusion_matrix[2,1]-(kmeans_confusion_matrix[1,1]+kmeans_confusion_matrix[2,0])<0:
                kmeans_confusion_matrix = kmeans_confusion_matrix[:,::-1]
                kmeans_centers = kmeans_centers[::-1,:]
                color0 = np.array([0.0, 1.0, 1.0, 0.1])
                color1 = np.array([1.0, 0.0, 1.0, 0.1])
            else:
                color1 = np.array([0.0, 1.0, 1.0, 0.1])
                color0 = np.array([1.0, 0.0, 1.0, 0.1])
           
            # calculate the mid points between the kmeans centroids
            kmeans_middle_point = np.mean(kmeans_centers,axis=0)
            # calculate the kmeans normal vector and normalize it
            kmeans_n = (kmeans_centers[0,:]-kmeans_middle_point).reshape(-1,1)
            
            # kmeans color trick
            y_ax_temp = line_eq(ax_temp_xlim,kmeans_centers)
            bbPath = mplPath.Path(np.array([[ax_temp_xlim[1], ax_temp_ylim[0]],
                                            [ax_temp_xlim[1], y_ax_temp[1]],
                                            [ax_temp_xlim[0], y_ax_temp[0]],
                                            [ax_temp_xlim[0], ax_temp_ylim[0]]]))
            if bbPath.contains_point(kmeans_centers[0])&(kmeans.predict(kmeans_centers[:1,:])==0):
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[0],color=color1,zorder=0)
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[1],color=color0,zorder=0)
                
                kmeans_n = (kmeans_centers[1,:]-kmeans_middle_point).reshape(-1,1)
            elif bbPath.contains_point(kmeans_centers[1])&(kmeans.predict(kmeans_centers[:1,:])==0):
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[0],color=color0,zorder=0)
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[1],color=color1,zorder=0)
                
                kmeans_n = (kmeans_centers[1,:]-kmeans_middle_point).reshape(-1,1)
            elif bbPath.contains_point(kmeans_centers[0])&(kmeans.predict(kmeans_centers[:1,:])==1):
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[0],color=color0,zorder=0)
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[1],color=color1,zorder=0)
            elif bbPath.contains_point(kmeans_centers[1])&(kmeans.predict(kmeans_centers[:1,:])==1):
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[0],color=color1,zorder=0)
                ax_temp.fill_between(ax_temp_xlim[::-1],y_ax_temp[::-1],ax_temp_ylim[1],color=color0,zorder=0)
            
            # normalize the kmeans normal vector
            kmeans_n/=np.linalg.norm(kmeans_n,ord=2)
            
            #export data
            if bool_exportdata:
                d_iter_temp['kmeans'] = kmeans

            if i_data==data_emb[-1]:#i_data in data_emb:                                
                # project on kmeans_n
                new_y_pred_emb = (((kmeans_n.T@y_pred_emb.T/(kmeans_n.T@kmeans_n))).reshape(-1,1)-kmeans_n.T@kmeans_middle_point/(kmeans_n.T@kmeans_n))
                
                # calculate covariance
                var = np.diag(np.exp(-distance_te_te/k_scale_class_opt)-np.exp(-distance_te_trtot/k_scale_class_opt)@np.linalg.inv(np.exp(-distance_trtot_trtot/k_scale_class_opt)+1e-8*np.eye(distance_trtot_trtot.shape[0]))@np.exp(-distance_te_trtot.T/k_scale_class_opt))
                
                # plot Layer 3, positive
                ax_layer3_0.scatter((new_y_pred_emb[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]>0.5,:])[:,0],var[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]>0.5],c=cmap_bad_1_2_3(y_raw_test.flatten()[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]>0.5]/3),marker='.', edgecolor='black', s=60,linewidth=0.1)
                ax_layer3_0.scatter((new_y_pred_emb[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]>0.5,:])[:,0],var[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]>0.5],c=cmap_bad_1_2_3(y_raw_test.flatten()[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]>0.5]/3),marker='.', edgecolor='black', s=60,linewidth=0.1)
                
                # plot Layer 3, negatives
                ax_layer3_1.scatter((new_y_pred_emb[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]<0.5,:])[:,0],var[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]<0.5],c=cmap_bad_1_2_3(y_raw_test.flatten()[y_raw_test.flatten()!=3][y_pred_test.flatten()[y_raw_test.flatten()!=3]<0.5]/3),marker='.', edgecolor='black', s=60,linewidth=0.1)
                ax_layer3_1.scatter((new_y_pred_emb[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]<0.5,:])[:,0],var[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]<0.5],c=cmap_bad_1_2_3(y_raw_test.flatten()[y_raw_test.flatten()==3][y_pred_test.flatten()[y_raw_test.flatten()==3]<0.5]/3),marker='.', edgecolor='black', s=60,linewidth=0.1)
                
                # log scale
                ax_layer3_0.set_yscale('log')
                ax_layer3_1.set_yscale('log')
                
                # set axes limits
                ax_layer3_0.set_xlim((np.min([ax_layer3_0.get_xlim()[0],ax_layer3_1.get_xlim()[0]]),
                                      np.max([ax_layer3_0.get_xlim()[1],ax_layer3_1.get_xlim()[1]]),
                                      ))
                ax_layer3_0.set_ylim((np.min([ax_layer3_0.get_ylim()[0],ax_layer3_1.get_ylim()[0]]),
                                      np.max([ax_layer3_0.get_ylim()[1],ax_layer3_1.get_ylim()[1]]),
                                      ))
                ax_layer3_1.set_xlim(ax_layer3_0.get_xlim())
                ax_layer3_1.set_ylim(ax_layer3_0.get_ylim())
                
                
                # color accordingly to k-means
                ax_layer3_0.fill_betweenx(np.array(ax_layer3_0.get_ylim()),ax_layer3_0.get_xlim()[0],0,color=color1,zorder=0)
                ax_layer3_0.fill_betweenx(np.array(ax_layer3_0.get_ylim()),0,ax_layer3_0.get_xlim()[1],color=color0,zorder=0)
                ax_layer3_1.fill_betweenx(np.array(ax_layer3_1.get_ylim()),ax_layer3_1.get_xlim()[0],0,color=color1,zorder=0)
                ax_layer3_1.fill_betweenx(np.array(ax_layer3_1.get_ylim()),0,ax_layer3_1.get_xlim()[1],color=color0,zorder=0)

                # Layer 3 (positive, embedding+covariance, k-means) confusion matrix
                temp_0 = np.unique(y_raw_test[y_pred_test.flatten()>0.5,:][new_y_pred_emb[y_pred_test.flatten()>0.5,0]>0,0],return_counts=True)
                temp = np.zeros((2,4));temp[0,:]=np.arange(4)
                temp[1,temp_0[0]] = temp_0[1]
                temp_0 = temp
                temp_1 = np.unique(y_raw_test[y_pred_test.flatten()>0.5,:][new_y_pred_emb[y_pred_test.flatten()>0.5,0]<0,0],return_counts=True)
                temp = np.zeros((2,4));temp[0,:]=np.arange(4)
                temp[1,temp_1[0]] = temp_1[1]
                temp_1 = temp
                print('\n')
                print('Layer 3 (positive, embedding+covariance, k-means) confusion matrix:')
                print(np.concatenate((temp_0,temp_1),axis=0)[[1,3],:].T)
                
                # Layer 3 (positive, embedding+covariance, k-means) confusion matrix
                temp_0 = np.unique(y_raw_test[y_pred_test.flatten()<0.5,:][new_y_pred_emb[y_pred_test.flatten()<0.5,0]>0,0],return_counts=True)
                temp = np.zeros((2,4));temp[0,:]=np.arange(4)
                temp[1,temp_0[0]] = temp_0[1]
                temp_0 = temp
                temp_1 = np.unique(y_raw_test[y_pred_test.flatten()<0.5,:][new_y_pred_emb[y_pred_test.flatten()<0.5,0]<0,0],return_counts=True)
                temp = np.zeros((2,4));temp[0,:]=np.arange(4)
                temp[1,temp_1[0]] = temp_1[1]
                temp_1 = temp
                print('Layer 3 (negatives, embedding+covariance, k-means) confusion matrix:')
                print(np.concatenate((temp_0,temp_1),axis=0)[[1,3],:].T)
                
                #export data
                if bool_exportdata:
                    d_iter_temp['embedding components projection (positive classified)'] = new_y_pred_emb[y_pred_test.flatten()>0.5,:]
                    d_iter_temp['embedding components projection (negative classified)'] = new_y_pred_emb[y_pred_test.flatten()<0.5,:]
                    d_iter_temp['posterior covariance (positive classified)'] = var[y_pred_test.flatten()>0.5]
                    d_iter_temp['posterior covariance (negative classified)'] = var[y_pred_test.flatten()<0.5]

        # interactive plot
        plt.pause(0.001)
        
        # export data
        if bool_exportdata:
            d_iter[f'iteration {i}'] = d_iter_temp
                
            
# finalize fraction of good curves found
frac_data.append(1)
frac_data = np.array(frac_data)
result_frac_positive.append(1)
result_frac_positive=np.array(result_frac_positive)

# sub-figure for Layer 1 (binary classifier)
ax_layer1.clear()
ax_layer1.plot([0,(y_raw>0).sum()/data.max(),1],[0,1,1],':k',label='perfect classifier')
ax_layer1.plot(frac_data,result_frac_positive,'k',label='FUSION')
ax_layer1.plot(frac_data,frac_data,'k--',label='random sampling')
ax_layer1.set_xlim(0,1)
ax_layer1.set_ylim(0,1)
if file_name_dataset == 'Simulation_test.npz':
    ax_layer1.set_xticks(np.concatenate((data_emb,np.array([len(x_emb_train)])))/len(x_emb_train))
    ax_layer1.set_xticklabels(np.concatenate((data_emb,np.array([len(x_emb_train)]))))
else:
    ax_layer1.set_xticks(data_emb/len(x_emb_train))
    ax_layer1.set_xticklabels(data_emb)    
ax_layer1.set_xticks(np.linspace(0,1,5),minor=True)
ax_layer1.tick_params(axis="x",direction="in",which='minor')
ax_layer1.tick_params(axis="x",direction="out",which='major')
ax_layer1.set_yticks(np.linspace(0,1,5))
ax_layer1.set_yticklabels((np.linspace(0,1,5)*100).astype(int))
ax_layer1.grid(axis='y',which='major',linestyle=':')
ax_layer1.grid(axis='x',which='minor',linestyle=':')
ax_layer1.grid(axis='x',which='major',linestyle='-.')
ax_layer1.legend(loc='lower right')

ax_layer1_sec_x = ax_layer1.secondary_xaxis('top')
ax_layer1_sec_y = ax_layer1.secondary_yaxis('right')
ax_layer1_sec_x.set_xticks(np.linspace(0,1,5))
ax_layer1_sec_x.set_xticklabels((np.linspace(0,1,5)*100).astype(int))
ax_layer1_sec_y.set_yticks(np.linspace(0,1,5))
ax_layer1_sec_y.set_yticklabels((np.linspace(0,1,5)*y.sum()).astype(int))

ax_layer1_sec_x.grid(axis='y')
ax_layer1_sec_x.grid(axis='x')
ax_layer1_sec_y.grid(axis='y')
ax_layer1_sec_y.grid(axis='x')

ax_layer1.set_xlabel('Number of curves analyzed')
ax_layer1.set_ylabel('$\%$ of good curves found')
ax_layer1_sec_x.set_xlabel('$\%$ of curves analyzed')
ax_layer1_sec_y.set_ylabel('Number of good curves found')

# sub-figure for Layer 2 (embedding space)
ax_layer2_0.set_ylabel('$y_{emb}$')
ax_layer2_0.set_xlabel('$x_{emb}$')
ax_layer2_1.set_xlabel('$x_{emb}$')
ax_layer2_2.set_xlabel('$x_{emb}$')

# sub-figure for Layer 3 (embedding space + covariance)
ax_layer3_0.set_xlabel('$n_{k-means}$')
ax_layer3_0.set_ylabel('Covariance')
ax_layer3_1.set_xlabel('$n_{k-means}$')
ax_layer3_1.set_ylabel('Covariance')
ax_layer3_1.yaxis.set_label_position("right")
ax_layer3_1.tick_params(which='both',right=True,labelright=True,left=False,labelleft=False)

# save figure
plt.tight_layout()
if bool_savefigs:
    plt.savefig(f'./Figures/{dataset}',dpi=600)

if bool_exportdata:
    from pathlib import Path
    Path("./ExportedData/").mkdir(parents=True, exist_ok=True)
    with open(f'./ExportedData/{dataset}.pkl', 'wb') as f:
        pickle.dump({'General Data':d_general,'Data Iterations':d_iter}, f)
