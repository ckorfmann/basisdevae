'''
BasisDeVAE Encoder class.

Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/encoder.py
Original author: Kaspar Märtens
'''

from torch import device, dtype, Tensor, float32
from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torch.nn.functional import softplus

class Encoder(Module):
    '''
    Encoder

    Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/encoder.py
    Original author: Kaspar Märtens
    '''

    def __init__(self, n_features: int, hidden_dim: int, z_dim: int,
                 nonlinearity: Module=ReLU,
                 device: device=device('cpu'),
                 dtype: dtype=float32,
                 model_name: str='Encoder'):
        '''
        Encoder for the VAE (neural network that maps P-dimensional data to [mu_z, sigma_z])

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/encoder.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        n_features : int
            Number of data features
        hidden_dim : int
            Number of output dimensions for hidden layers
        z_dim : int
            Number of latent space dimensions
        nonlinearity : torch.nn.Module, optional
            Activation to be used in hidden layers, by default `torch.nn.ReLU`
        device : str, optional
            Pytorch device to be used, by default 'cpu'
        model_name: str, optional
            Name of module by default 'Encoder'
        '''

        super().__init__()
        self.device = device
        self.dtype = dtype

        self.tensor_params = {
            'device' : self.device,
            'dtype' : self.dtype,
        }
    
        self.n_features = n_features
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.hidden_dim = hidden_dim
        self.model_name = model_name

        self.__build_network()

    def forward(self, y) -> tuple[Tensor,Tensor]:
        '''
        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/encoder.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        y : _type_
            _description_

        Returns
        -------
        tuple[Tensor,Tensor]
            _description_
        '''
        network_outputs = self.network(y)

        mu_z = network_outputs[ : , : self.z_dim ]
        sigma_z = 1e-6 + softplus(network_outputs[ : , self.z_dim : 2 * self.z_dim ])

        return mu_z, sigma_z

    def __build_network(self):
        '''
        Builds Encoder network.

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/encoder.py
        Original author: Kaspar Märtens
        '''

        self.linear1 = Linear(self.n_features, self.hidden_dim, **self.tensor_params)
        self.linear2 = Linear(self.hidden_dim, 2 * self.z_dim, **self.tensor_params)

        self.network = Sequential(
            self.linear1,
            self.nonlinearity(),
            self.linear2,
        )
