import matplotlib.pyplot as plt
# import numpy as np

from numpy import dtype, ndarray, float32, exp, hstack, linspace, log, savetxt, stack
from numpy.random import uniform
from scipy.stats import norm

def generate_gaussian_samples(num_samples: int, num_gaussians: int,
                              min_y: float, max_y: float,
                              min_x: float, max_x: float,
                              dtype: dtype=float32) -> ndarray:
    x = linspace(min_x, max_x, num_samples)

    gaussian_list = list()
    for _ in range(num_gaussians):
        output_scale = uniform(min_y, max_y)

        pdf_loc = uniform(low=-2.5 / x.std(),
                                    high=2.5 / x.std())
        pdf_scale = uniform(low=x.std() / 2,
                                      high=2 / x.std())

        pdf = norm.pdf(x, loc=pdf_loc, scale=pdf_scale)
        pdf = output_scale * pdf / pdf.max()

        gaussian_list.append(pdf)

    return stack(gaussian_list, axis=1, dtype=dtype)

def generate_monotonic_samples(num_samples: int, num_monotonics: int,
                               min_y: float, max_y: float,
                               min_x: float, max_x: float,
                               ratio: float=0.5,
                               dtype: dtype=float32) -> ndarray:
    x = linspace(0, abs(min_x) + abs(max_x), num_samples)

    func_list = list()
    for i in range(num_monotonics):
        x_translation = uniform(x.std()/1.25, x.var()*1.1)
        
        y_scale = uniform(min_y if i < num_monotonics * ratio else 0.3,
                                    max_y if i >= num_monotonics * ratio else -0.3)

        y = log(1 + exp(x - x_translation))
        y = y_scale * y / abs(y).max()

        func_list.append(y)

    return stack(func_list, axis=1, dtype=dtype)

def plot_examples(examples: list, colors: list, x: ndarray=None, min_x: float=-3, max_x: float=3, title: str='Synthetic Data Examples', save: bool=False, fname: str='./datset_sample.png') -> None:
    num_samples = examples[0].shape[0]
    if x == None:
        x = linspace(min_x, max_x, num_samples)
    
    plt.figure()
    for i, example in enumerate(examples):
        plt.plot(x, example, c=colors[i])
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('x(t)')

    if save:
        plt.savefig(fname, bbox_inches='tight')
        return
    
    plt.show()

def generate_data(n_samples: int=500, n_features: int=50,
                  gaussian_ratio: float=0.6,
                  min_x: float=-3, max_x: float=3,
                  gaus_min_y: float=0.5, gaus_max_y: float=1.7,
                  mono_min_y: float=-2.5, mono_max_y: float=2.5,
                  save_to_file: bool=False,
                  fname: str='dataset.csv',
                  savedir: str= './',
                  dtype: dtype=float32) -> ndarray:
    
    num_monotonics = int((1 - gaussian_ratio) * n_features)
    num_gaussians = n_features - num_monotonics

    gaussians = generate_gaussian_samples(n_samples, num_gaussians, min_y=gaus_min_y, max_y=gaus_max_y, min_x=min_x, max_x=max_x, dtype=dtype)
    monotonics = generate_monotonic_samples(n_samples, num_monotonics, min_y=mono_min_y, max_y=mono_max_y, min_x=min_x, max_x=max_x, dtype=dtype)

    data = hstack([gaussians, monotonics], dtype=dtype)

    if save_to_file:
        savetxt(f'{savedir}/{fname}', data, delimiter=',', fmt='%f')

    return data
