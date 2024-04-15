'''
Generates synthetic data for training and testing deep learning models.

`synth_data_gen.py` is adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py
Original author: Dominic Danks
'''

import matplotlib.pyplot as plt
import numpy as np


def gaussian(A: np.ndarray, t0: np.ndarray, sigma: np.ndarray, t: np.ndarray, dtype: np.dtype=np.float32) -> np.ndarray:
    '''
    Generates batched Gaussian distributions.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py

    Parameters
    ----------
    A : Numpy.ndarray
        shape=(d,)
    t0 : Numpy.ndarray
        shape=(d,)
    sigma : Numpy.ndarray
        shape=(d,)    
    t : Numpy.ndarray
        shape=(N,)

    Returns
    -------
    Numpy.ndarray
        (N,d) numpy array
    '''

    t_ = t[ : , None ]
    t0_ = t0[ None , : ]
    A_ = A[ None , : ]

    return (A_ * np.exp(-0.5 * (t_ - t0_)**2 / sigma**2)).astype(dtype)

def scaled_softplus(t: np.ndarray, t_shift: np.ndarray, A: np.ndarray, dtype: np.dtype=np.float32) -> np.ndarray:
    '''
    Generates batched scaled softplus distributions.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py

    Parameters
    ----------
    t : Numpy.ndarray
        shape=(d,)
    t_shift : Numpy.ndarray
        shape=(d,)
    A : Numpy.ndarray
        shape=(N,)

    Returns
    -------
    Numpy.ndarray
        (N,d) numpy array
    '''

    return (A[ None , : ] * 0.4 * np.logaddexp(0, t[ : , None ] + t_shift[ None , : ])).astype(dtype)

def _generate_gaussian_components(t: np.ndarray, n_dim: int, plot_example: bool=False, dtype: np.dtype=np.float32) -> np.ndarray:
    '''
    Generates synthetic data that follows a 
    Gaussian distribution.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py

    Parameters
    ----------
    t : np.ndarray
        Timescale
    plot_example : bool, optional
        Plot example of generated data, by default False

    Returns
    -------
    np.ndarray
        Generated data.
    '''

    A = np.linspace(0.4, 1.6, n_dim, dtype=dtype)
    t0 = np.linspace(-1.5, 1.5, n_dim, dtype=dtype)

    np.random.shuffle(A)
    np.random.shuffle(t0)

    sigma = np.ones(n_dim, dtype=dtype)
    gaussians = gaussian(A, t0, sigma, t, dtype=dtype)
    
    if plot_example:
        plt.figure()
        plt.plot(gaussians[0,...])
        plt.title('Gaussian feature')

    return gaussians.astype(dtype)

def _generate_increasing_components(t: np.ndarray, n_dim: int, plot_example: bool=False, dtype: np.dtype=np.float32) -> np.ndarray:
    '''
    Generates synthetic data that is monotonically increasing.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py

    Parameters
    ----------
    t : np.ndarray
        Timescale
    plot_example : bool, optional
        Plot example of generated data, by default False

    Returns
    -------
    np.ndarray
        Generated data.
    '''

    t_shift_inc = np.linspace(-1.5, 1.5, n_dim, dtype=dtype)
    A_inc = np.linspace(0.4, 1.6, n_dim, dtype=dtype)

    np.random.shuffle(A_inc)
    np.random.shuffle(t_shift_inc)

    monoinc = scaled_softplus(t, t_shift_inc, A_inc, dtype=dtype)

    if plot_example:
        plt.figure()
        plt.plot(monoinc[0,...])
        plt.title('Increasing feature')

    return monoinc.astype(dtype)

def _generate_decreasing_components(t: np.ndarray, n_dim: int, plot_example: bool=False, dtype: np.dtype=np.float32) -> np.ndarray:
    '''
    Generates synthetic data that is monotonically decreasing.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py

    Parameters
    ----------
    t : np.ndarray
        Timescale
    plot_example : bool, optional
        Plot example of generated data, by default False

    Returns
    -------
    np.ndarray
        Generated data.
    '''

    t_shift_dec = np.linspace(-1.5, 1.5, n_dim, dtype=dtype)
    A_dec = np.linspace(0.4, 1.6, n_dim, dtype=dtype)

    np.random.shuffle(A_dec)
    np.random.shuffle(t_shift_dec)

    monodec = -scaled_softplus(t, t_shift_dec, A_dec, dtype=dtype)

    if plot_example:
        plt.figure()
        plt.plot(monodec[0,...])
        plt.title('Decreasing feature')

    return monodec.astype(dtype)

def _plot_examples(t, examples: list, colors: list=None, save: bool=False, fname: str='./synth_features.png') -> None:
    '''
    Plots generated data.

    Parameters
    ----------
    t : np.ndarray
        Timescale
    examples : list
        Generated data
    colors : list, optional
        Color to be used for plot lines, by 
        default None
    save : bool, optional
        Save plot to file, by default False
    out_fname : str, optional
        Filename used for saving plot if `save` 
        is True, by default 'synth_features.png'
    '''

    plt.figure()
    
    for i, example in enumerate(examples):
        plt.plot(
            t,
            example,
            c=None if colors == None else colors[i],
        )

    plt.title('Synthetic Data Examples')
    plt.xlabel('t')
    plt.ylabel('x(t)')

    if save:
        plt.savefig(fname, bbox_inches='tight')
        return
    
    plt.show()

def generate_data(n_samples: int=500, n_features: int=50, trans_ratio: float=0.6, seed: int=None, plot_example: bool=False, save_to_file: bool=False, fname: str='./synth_features.png', dtype: np.dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates synthetic data.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/synth_data_gen.py

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to be generated, by default 500
    seed : int, optional
        Seed used for Numpy.random, by default None
    plot_example : bool, optional
        Plot generated data, by default False
    save_to_file : bool, optional
        Save generated data to file, by default False

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[float]]
        (timescale, generated data)
    '''
    
    if not seed == None:
        assert type(seed)  == int, f'Invalid seed "{seed}"'
        np.random.seed(seed)

    monotonic_dim = int(((1 - trans_ratio) * n_features) / 2)
    transient_dim = n_features - monotonic_dim * 2


    t = np.linspace(-3, 3, n_samples, dtype) # Times considered (ground truth pseudotimes)

    gaussians = _generate_gaussian_components(t, transient_dim)
    monoinc = _generate_increasing_components(t, monotonic_dim)
    monodec = _generate_decreasing_components(t, monotonic_dim)

    if plot_example:
        _plot_examples(
            t,
            [gaussians, monoinc, monodec],
            ['red', 'green', 'blue'],
            save=plot_example,
        )

    dataset = np.concatenate((gaussians, monoinc, monodec), axis=1)

    data_dim_idx = 1
    assert dataset.shape[data_dim_idx] == n_features, f'Invalid dataset shape {dataset.shape[data_dim_idx]}'

    if save_to_file:
        np.savetxt('synth_x.csv', dataset, delimiter=',', fmt='%f')
        np.savetxt('synth_t.csv', t, delimiter=',', fmt='%f')

    return t, dataset
