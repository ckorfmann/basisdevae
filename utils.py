import numpy as np
import torch

from copy import deepcopy
from itertools import combinations
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.random import randn, shuffle
from os import makedirs
from os.path import exists
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import TensorDataset, DataLoader

from data_generation import plot_examples
from VAE import build_VAE, get_cluster_assignments, train_model

def add_noise(data: ndarray, noise_level: float, dtype: np.dtype=np.float32) -> ndarray:
    '''
    Add random noise to `data`

    Parameters
    ----------
    data : NDArray
        Data array
    noise_level : float
        Amount of noise
    dtype : np.dtype, optional
        Output dtype, by default `np.float32`

    Returns
    -------
    NDArray
        Noisy data
    '''

    return (data + noise_level * randn(*data.shape)).astype(dtype)

def copy_data_into_tensors(data: dict[str,ndarray], device: torch.device=torch.device('cpu'), dtype: torch.dtype=torch.float32) -> dict[str,TensorDataset]:
    '''
    Copy Numpy arrays from `data` into TensorDatasets.

    Parameters
    ----------
    data : dict[str,ndarray]
        Dictionary containing Numpy arrays
    device : torch.device, optional
        Pytorch device, by default `torch.device('cpu')`
    dtype : torch.dtype, optional
        Pytorch dtype, by default `torch.float32`

    Returns
    -------
    dict[str,TensorDataset]
        Dictionary of TensorDatasets, uses the same 
        keys as `data`
    '''

    return { name : TensorDataset(torch.from_numpy(deepcopy(d)).type(dtype).to(device=device))
                for name, d
                in data.items() }

def generate_dataloaders(datasets: dict[str,TensorDataset], batch_size: int, shuffle_datasets: bool=True, shuffle_dataloaders: bool=True) -> dict[str,DataLoader]:
    '''
    Copy TensorDatasets from `datasets` into DataLoader.

    Parameters
    ----------
    datasets : dict[str,TensorDataset]
        Dictionary containing TensorDatasets.
    dataloader_params : dict[str], optional
        Parameters to pass to the TensorDataset 
        initializer, by default `None`.
    shuffle_datasets: bool, optiona
        Shuffle data within datasets, by 
        default `True`.
    shuffle_dataloaders: bool=True
        Shuffle order of dataloaders in 
        returned dict, by default `True`.
        
    Returns
    -------
    dict[str,DataLoader]
        Dictionary of DataLoaders, uses the same 
        keys as `data`
    '''

    if shuffle_dataloaders:
        datasets = list(datasets.items())
        shuffle(datasets)

    return { name : DataLoader(ds, batch_size=batch_size, shuffle=shuffle_datasets)
                for name, ds
                in datasets }

def train_clusterings_on_datasets(dataloaders: dict[str,DataLoader], model_params: dict[str], epochs: int, dtype: torch.dtype=None, save: bool=False, fname: str=None, save_dir: str='') -> dict[str,ndarray]:
    '''
    Builds and trains VAE models on datasets in `dataloaders` 
    to obtain clustering assignments.

    Parameters
    ----------
    dataloaders : dict[str,DataLoader]
        Dictionary containing DataLoaders.
    model_params : dict[str]
        Model parameters required for Encoder, Decoder, 
        and full VAE model.
    train_epochs : int
        Number of training epochs.
    dtype : torch.dtype, optional
        Pytorch dtype to cast datasets. If `None`, will 
        infer from `dataloaders`, by default `None`.
    save : bool, optional
        Save model weights to disk, by default `False`.
    save_dir: str, optional
        Directory to save model weights if `save` is 
        `True`, by default ''.

    Returns
    -------
    dict[str,ndarray]
        Clustering assignments for trained models. Keys 
        are same as `dataloaders` keys.
    '''

    if dtype == None:
        dtype = list(dataloaders.values())[0].dataset.tensors[0].dtype

    clustering_assignments = dict()
    for name, ds in dataloaders.items():
        print(f'Dataset "{name}"')

        # x0init_ = ds.dataset.tensors[0].mean(dim=0, keepdim=True).type(dtype)
        model = build_VAE(model_params)

        train_model(model=model,
                    data_loader=ds,
                    ds_name=name,
                    epochs=epochs,
                    save=save,
                    fname_base=fname,
                    save_dir=save_dir)
        
        clustering_assignments[name] = get_cluster_assignments(model)

        if save and name == 'noisy':
            eval_model = model.eval()

            idx = np.arange(model.decoder.n_features)

            z_list = list()
            Y_pred_list = list()

            for (Y_subset, ) in ds:
                mu_, sigma_ = eval_model.encoder(
                    Y_subset.type(model.decoder.dtype).to(eval_model.device)
                )
                z_mu = mu_.detach().cpu().numpy().squeeze()
                
                Y_ = eval_model.decoder(
                    torch.tensor(z_mu, dtype=model.decoder.dtype, device=eval_model.device)[ : , None ]
                ).detach().cpu().numpy()[ : , idx , clustering_assignments[name] ]

                Y_pred_list.append(Y_)
                z_list.append(z_mu)

            z = np.hstack(z_list)
            Y_pred = np.vstack(Y_pred_list)

            z_argsort = z.argsort()

            z_grid = z[z_argsort]
            Y_pred = Y_pred[ z_argsort , : ]

            plt.rcParams.update({'font.size': 16})
            plt.figure()

            colors = [
                'green',
                'blue',
                'red', 
            ]
            for j in range(model.decoder.n_features):
                plt.plot(
                    z_grid,
                    Y_pred[ : , j ],
                    c=colors[clustering_assignments[name][j]],
                )
            plt.xlabel('z')
            plt.ylabel('x(z)')
            plt.title('Denoised outputs')
            plt.tight_layout()
            plt.savefig(
                f"./denoised_outputs.png",
                bbox_inches='tight',
            )

    return clustering_assignments

def get_scores(clustering_assignments: dict[str,ndarray], subsets: list[str], gt: str=None) -> dict[str,list[float]]:
    '''
    Get ARI scores for clustering assignments.

    Parameters
    ----------
    clustering_assignments : dict[str,ndarray]
        Dictionary of clustering assignments.
    subsets : list[str]
        Keys of sub-datasets to be scored.
    gt : str, optional
        Ground truth key, by default `None`.

    Returns
    ----------
    dict[str,list[float]]
        ARI scores for indicated subsets.
    '''

    scores = dict()

    if not gt == None:
        for subset in subsets:
            scores[f'{gt}_{subset}'] = adjusted_rand_score(clustering_assignments[gt], clustering_assignments[subset])

    for (subset_a, subset_b) in combinations(subsets, 2):
        scores[f'{subset_a}_{subset_b}'] = adjusted_rand_score(clustering_assignments[subset_a], clustering_assignments[subset_b])

    return scores

def save_scores(scores: dict[str,ndarray], save_dir: str='./', fname: str='scores.csv') -> None:
    '''
    Saves ARI scores to disk.

    Parameters
    ----------
    scores : NDArray
        Array of ARI scores.
    save_dir : str, optional
        Output directory, by default './'
    fname : str, optional
        Output filename, by default 'scores.csv'.
    '''

    clustering_scores_repr = [ ','.join([name, *[str(a) for a in score]])
                                for name, score
                                in scores.items() ]
    
    if not exists(save_dir):
        makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{fname}', 'w') as fout:
        print(
            *clustering_scores_repr,
            sep='\n',
            file=fout,
        )

def print_stats(scores: dict[str,ndarray]) -> None:
    '''
    Prints mean ARI score, score standard 
    deviation, min score, and max score 
    to stdout.

    Parameters
    ----------
    scores : dict[str,ndarray]
        ARI scores.
    '''

    for k, score in scores.items():
        a, b = k.split('_')
        
        print(
            f'ARI({a}, {b}):',
            f'\tMean = {np.mean(score):1.3f}',
            f'\tStd =  {np.std(score):1.3f}',
            f'\tMin =  {min(score):1.3f}',
            f'\tMax =  {max(score):1.3f}',
            sep='\n',
            end='\n\n',
        )

def save_boxplot(scores: dict[str,ndarray], fname: str='boxplot', save_dir: str='./', save_format: str='png', usermedians: list=None) -> None:
    if not exists(save_dir):
        makedirs(save_dir, exist_ok=True)

    # split keys into the following format:
    # 'gt_hi' -> 'ARI(gt, hi)'
    labels = [ f'ARI({k_split[0]}, {k_split[1]})'
                    for k
                    in scores.keys()
                    if (k_split := k.split('_')) ]
    
    separation_ = 2
    positions_ = range(separation_, separation_ * len(scores) + 1, separation_)

    fig, ax = plt.subplots()
    VP = ax.boxplot(scores.values(),
                    positions=positions_, widths=separation_*0.75,
                    patch_artist=True, showfliers=True,
                    medianprops={"color": "red",
                                 "linewidth": 2,
                                 "linestyle": '--'},
                    usermedians=usermedians,
                    boxprops={"facecolor": "gray",
                              "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "gray",
                                  "linewidth": 1.5},
                    capprops={"color": "gray",
                              "linewidth": 1.5})
    
    ax.set_xticklabels(labels)
    ax.set_ylabel('ARI Score')
    plt.grid(True)

    fig.savefig(
        f'{save_dir}/{fname}.{save_format}',
        bbox_inches='tight',
    )
