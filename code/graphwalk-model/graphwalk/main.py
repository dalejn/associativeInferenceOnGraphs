
import numpy as np
import pandas as pd
from datetime import datetime
#import os # TODO: update paaths to path constructor

#from tqdm import tqdm # TODO: fix verbose training prints to logger or training bar
import networkx as nx

import torch
from torchsummary import summary # TODO: This should be a param / file write

from sys import path
path.append('/Users/dalejn/PycharmProjects/graphwalk_representation/graphwalk-model/graphwalk''')
# TODO: refactor into one nice callable API? ie.:
# import graphwalk as gw
# gw.train; gw.make_graph, gw.learn, gw.plot ?
from graphtask import *
from graphmeta import mappingN, Gedges
from graphtrain import Data, TrainTorch, get_graph_dataset
from graphplots import plot_graphtask
from learner import AE, get_hidden_activations
from utils import calc_dist


### THE GRAPH TASK 
G = nx.from_numpy_array(Gedges)
edges = np.array(list(G.edges)) 
if 0: plot_graphtask(G, mappingN, Gedges, font_size=10) # TODO: I don't think there's a show call

# Set device and params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {
    'num_epochs': 10,
    'learning_rate': 1e-2,
    'weight_decay': 1e-5,
    'device': device
}
data_dir = '/Users/dalejn/Desktop/graph_representation/weights_new' # os.path.join()
save_results = True
save_weights = True

n_models = 100
hidden_layer_widths = [6, 9, 12, 15, 18]
results = {'name':[], 'path':[], 'task':[], 'L2':[], '1':[], '2':[], '3':[], '4':[],
            'scores':[], 'end_loss':[], 'hidden':[], 'dists':[]}

# Sweet jesus this tripple for loop makes me sad
for dataset_ID in ['I', 'B']:
    for l2_size in hidden_layer_widths:
        for model_i in range(n_models):
            print(f'Training {dataset_ID} model: {model_i} L2 {l2_size}')

            # build dataset...
            X, Y = get_graph_dataset(edges, sel=dataset_ID)
            dat = Data(X, Y, batch_size=8, datatype=dataset_ID, shuffle=False)

            # Set model metadata and create model and trainer
            model_name = f'{dataset_ID}_{l2_size}_{model_i}'
            weight_path = f'{data_dir}/torchweights/{model_name}.pt' # os.path.join()
            model = AE(input_shape=dat.data_shape, L1=12, L2=l2_size, n_hidden=3, 
                        name=f'{model_name}', weight_path=weight_path).to(device)
            if 0: summary(model, input_size=(1,20))
            net = TrainTorch(model, params)

            # Train model 
            if dat.datatype == 'I': net.train(dat.dataloader)
            elif dat.datatype == 'B': net.train_blocked(dat.dataloaders)
            else:
                raise ValueError(f'datatype trainer is {dat.datatype}')

            # save model weights
            if save_weights: torch.save(net.model.state_dict(), weight_path)
            
            # Compute hidden activations and task distances
            hiddenarr = get_hidden_activations(net.model, n_hidden=3, 
                                                n_items=12, device=device)
            model_dists = calc_dist(hiddenarr, hiddenarr)
            path_lens = nx.floyd_warshall_numpy(G)
            trialIter = 500

            # Compute choice task results
            choice_accs_dist = relative_distance(12, model_dists, path_lens, 
                                                    ndistTrials=trialIter, verbose=False)
            dist_pct = {} # TODO: Wrap into function
            for dist, vals in choice_accs_dist.items():
                acc = (np.sum(vals) / len(vals)) * 100
                dist_pct[dist] = acc
                if 0: print(f'{dist}: {acc:.2f}% {len(vals)}')

            # Pack up into dictionary 
            # TODO: this should be an interable
            results['name'].append(model_name)
            results['path'].append(weight_path)
            results['task'].append(dataset_ID)
            results['L2'].append(l2_size)
            results['1'].append(dist_pct[1])
            results['2'].append(dist_pct[2])
            results['3'].append(dist_pct[3])
            results['4'].append(dist_pct[4])
            results['scores'].append(dist_pct)
            results['end_loss'].append(net.training_loss[-1])
            results['hidden'].append(hiddenarr)
            results['dists'].append(model_dists)


# Save results
r_frame = pd.DataFrame(results)

# Save dataframe
if save_results: 
    now = datetime.now().strftime('%m%d%H%M')
    frame_path = f'{data_dir}/saved_frames/{now}_dataframe.pkl' # os.path.join()
    r_frame.to_pickle(frame_path)
    r_frame.to_csv(f'{data_dir}/saved_frames/{now}_dataframe.csv')