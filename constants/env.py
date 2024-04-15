'''
Runtime environment parameters.

    `DEVICE`: torch.Device
        Global Pytorch device

    `MODEL_DIR`: str
        Output directory for model weights/state_dicts

    `NUMPY_DTYPE`: numpy.dtype
        Global Numpy dtype

    `RESULTS_DIR`: str
        Output directory for results

    `TORCH_DTYPE`: torch.dtype
        Global Pytorch dtype
'''

from numpy import dtype as numpy_dtype, float32 as numpy_float32
from torch import device, dtype as torch_dtype, float32 as torch_float32
from torch.cuda import is_available


DEVICE: device = device('cpu' if is_available() else 'cpu')
'''Global Pytorch device'''

MODEL_DIR: str = './models'
'''Output directory for model weights/state_dicts'''

NUMPY_DTYPE: numpy_dtype = numpy_float32
'''Global Numpy dtype'''

RESULTS_DIR: str = './results'
'''Output directory for results'''

TORCH_DTYPE: torch_dtype = torch_float32
'''Global Pytorch dtype'''
