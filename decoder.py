'''
BasisDeVAE Decoder class.

Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
Original author: Kaspar Märtens
'''

from torch import (device, dtype, Tensor,
                   float32,
                   cat, exp, hstack, 
                   lgamma, log, 
                   matmul, mean, ones,
                   ones_like, randn,
                   softmax, tensor, zeros)
from numpy.polynomial import legendre
from torch.distributions import gamma, normal
from torch.nn import Linear, Module, Parameter, Sequential, Softplus, Tanh
from torch.nn.init import calculate_gain, xavier_uniform_
from torch.nn.functional import softplus

class BasisODEDecoder(Module):
    '''
    BasisDeVAE decoder

    Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
    Original author: Kaspar Märtens
    '''

    def __init__(self,
                 n_features: int, hidden_dim: int, z_dim: int, n_basis: int,
                 alpha: float=1.0,
                 nonlinearity: Module=Tanh,
                 x0init=None,
                 quadrature_order: int=15,
                 device: device=device('cpu'),
                 dtype: dtype=float32,
                 model_name: str='Decoder'):
        '''
        BasisDeVAE decoder

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        n_features : int
            Number of features in data
        hidden_dim : int
            Number of feature dimensions to be used in network's hidden layers
        z_dim : int
            Number of latent space dimensions
        n_basis : int
            Number of basis functions
        nonlinearity : torch.nn.Module, optional
            Activation function to be used between network's hidden layers, 
            by default `torch.nn.Softplus`
        quadrature_order : int, optional
            Quadrature order used to evaluate integration operation, by 
            default `15`
        device : torch.device, optional
            Pytorch device to be used, by default `torch.device('cpu')`
        dtype : torch.dtype, optional
            Pytorch datatye to be used, by default `torch.float32`
        model_name : str, optional
            Name of module, by default 'Decoder'
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
        self.n_basis = n_basis
        self.output_dim = n_features * n_basis
        self.quadrature_order = quadrature_order
        self.x0init = x0init

        self.alpha_z = alpha * ones(n_basis, **self.tensor_params)
        self.lgamma_alpha = lgamma(self.alpha_z)
        self.term1 = lgamma(self.alpha_z.sum()) - lgamma(self.alpha_z.sum() + self.n_features)  # categorical_posterior

        self.__define_parameters()
        self.__calculate_quadrature()
        self.__build_network()


    def forward(self, z: Tensor):
        # [batch_size, n_features, n_basis]
        g0 = self.__network_passthrough(z).reshape((z.shape[0], self.n_basis, self.n_features)).swapdims(1,2)

        # [batch_size, n_basis, n_features]
        monotonic_basis = self.x0[ ... , None ] + g0[ ... , : 2 ]
       
        # [batch_size, n_features]   
        gaussian_basis = self.__get_gaussian_basis(g0, z)
        
        # [batch_size, n_features, n_basis]
        return self.__stack_basis(monotonic_basis, gaussian_basis)

    def loss(self, y_obs: Tensor, y_pred: Tensor, batch_scale: float=1.0, beta: float=1.0) -> float:
        '''
        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        inputs : torch.Tensor
            _description_
        prediction : torch.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        return - (batch_scale * self.__log_likelihood(y_obs, y_pred)) + beta * self.__ELBO() + self.__penalty()

    def get_phi(self) -> Tensor:
        '''
        Get clustering assignments using softmax

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens

        shape = `[n_features, n_basis]`
        
        Returns
        -------
        torch.Tensor
            Clustering assignments
        '''
        return softmax(self.qphi_logits, dim=-1)

    def set_x0_(self, x0: Tensor=None) -> None:
        self.x0 = Parameter(x0.to(device=self.device))

    def __build_network(self) -> None:
        '''
        Builds Decoder network.

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens
        '''

        self.linear1 = Linear(self.z_dim, self.hidden_dim, **self.tensor_params)
        self.linear2 = Linear(self.hidden_dim, self.output_dim, **self.tensor_params)

        self.network = Sequential(
            self.linear1,
            self.nonlinearity(),
            # Dropout(0.3),
            self.linear2,
            Softplus(),
        )
        self.network.apply(self.__init_weights)

    def __calculate_quadrature(self) -> Tensor:
        # [1, quadrature_order], [1, quadrature_order]
        leggauss_x, leggauss_y = legendre.leggauss(self.quadrature_order)
        self.u_i = tensor(leggauss_x[ None , : ], **self.tensor_params)
        self.w_i = tensor(leggauss_y[ None , : ], **self.tensor_params)

    def __calculate_z_translate(self, z: Tensor) -> Tensor:
        '''
        Gets z translation for Gaussian basis

        shape = `[batch_size * quadrature_order, n_features]`

        Parameters
        ----------
        z : torch.Tensor
            Latent space mapping.

        Returns
        -------
        torch.Tensor
        '''
        return z - self.zi[ ... , 0 ]

    def __define_parameters(self) -> None:
        '''
        Defines model parameters.

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens
        '''
        self.Aj = Parameter(xavier_uniform_(zeros([self.n_features, 1], **self.tensor_params),
                                            gain=calculate_gain('linear')))
        self.c = Parameter(zeros(self.n_features, **self.tensor_params))

        self.qphi_logits = Parameter(ones([self.n_features, self.n_basis], **self.tensor_params))
        self.sigma = Parameter(-2.0 * ones(self.n_features, **self.tensor_params))
        
        # Gaussian shift
        self.zi = Parameter(zeros([self.n_features, 1], **self.tensor_params))
        # max Gaussian point
        self.z0 = Parameter(xavier_uniform_(zeros([self.n_features, 1], **self.tensor_params),
                                            gain=calculate_gain('linear')))
        self.x0 = Parameter(randn([1, self.n_features], **self.tensor_params))

        if self.x0init is not None:
            self.x0 = Parameter(tensor(self.x0init, **self.tensor_params))

    def __ELBO(self) -> Tensor:
        '''
        BasisVAE.helpers.ELBO_collapsed_Categorical

        Returns
        -------
        torch.Tensor
            _description_
        '''
        phi = self.get_phi()
        n_k = phi.sum(dim=0)
        
        term2 = (lgamma(self.alpha_z + n_k) - self.lgamma_alpha).sum() # equation 4
        E_q_logq = (phi * log(phi + 1e-16)).sum() # equation 5

        return -self.term1 - term2 + E_q_logq

    def __get_Aj(self) -> Tensor:
        '''
        Returns the the the Gaussian basis Aj parameter while ensuring 
        it is greater than zero.

        shape = `[n_features, 1]`

        Returns
        -------
        torch.Tensor
            Network's Aj parameter
        '''
        return softplus(self.Aj)

    def __get_gaussian_basis(self, g0: Tensor, z: Tensor) -> Tensor:
        z_translate = self.__calculate_z_translate(z)

        h_j3 = z_translate * softplus(g0[ ... , 2 ] + self.c)
        return self.__get_Aj() * exp( - h_j3[ ... , None ] ** 2 )

    def __init_weights(self, m: Module) -> None:
        if isinstance(m, Linear):
            m.bias.data.fill_(0)

    def __log_likelihood(self, y_obs: Tensor, y_pred: Tensor):
        '''
        Log-likelihood

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        y_obs : torch.Tensor
            _description_
        y_pred : torch.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        '''

        sigma = 1e-4 + softplus(self.sigma)[ ... , None ]
        p_data = normal.Normal(y_pred, sigma)
        log_p = p_data.log_prob(y_obs[ ... , None ])

        phi = self.get_phi()[ None , ... ]

        return (phi * log_p).sum()

    def __network_passthrough(self, z: Tensor) -> Tensor:
        # [batch_size, quadrature_order]
        f_theta = matmul(z/2, 1 + self.u_i).flatten()[ ... , None ]

        # [batch_size, quadrature_order, output_dim]
        network_output = self.network(f_theta)

        z_translate = self.__calculate_z_translate(f_theta)

        stacked_outputs = self.__stack_network_output(network_output, z_translate).reshape((z.shape[0], self.u_i.shape[1], self.output_dim))
        
        # [batch_size, output_dim]
        return z/2 * (self.w_i[ ... , None ] * stacked_outputs).sum(dim=1)

    def __penalty(self):
        '''
        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
        Original author: Kaspar Märtens

        Returns
        -------
        _type_
            _description_
        '''
        A_j = self.__get_Aj()
        penalty = self.zi.pow(2).sum() + \
                    gamma.Gamma(ones_like(A_j), ones_like(A_j)).log_prob(A_j).sum()
        return penalty

    def __stack_basis(self, monotonic_basis: Tensor, gaussian_basis: Tensor) -> Tensor:
        return cat([monotonic_basis, gaussian_basis], dim=-1)

    def __stack_network_output(self, network_output: Tensor, z_translate: Tensor=None) -> Tensor:
        output_1 = network_output[ ... , : self.n_features ]
        output_2 = -network_output[ ... , self.n_features : 2 * self.n_features ]
        output_3 = network_output[ ... , -self.n_features : ] * z_translate

        return hstack([output_1, output_2, output_3])
