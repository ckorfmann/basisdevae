'''
BasisDeVAE Autoencoder class.

Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/VAE.py
Original author: Kaspar Märtens
'''

import numpy as np
from os import makedirs
from os.path import exists
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import BasisODEDecoder

class VAE(nn.Module):
    '''
    BasisDeVAE class

    Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/VAE.py
    Original author: Kaspar Märtens
    '''

    def __init__(self, encoder: Encoder, decoder: BasisODEDecoder, lr: float,
                 dtype: torch.dtype=torch.float32,
                 device: torch.device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.tensor_params = {
            'device' : self.device,
            'dtype' : self.dtype,
        }

        self.encoder = encoder
        self.decoder = decoder
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, y: torch.Tensor, batch_scale: float=1.0, beta: float=1.0) -> tuple[tuple[torch.Tensor,torch.Tensor,torch.Tensor],float]:
        '''
        Forward model passthrough.

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/VAE.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        y : torch.Tensor
            Input data tensor

        Returns
        -------
        tuple
            (z, mu_z, sigma_z), total_loss
        '''

        if self.decoder.x0 == None:
            print('yeet')
            self.decoder.set_x0_()

        mu_z, sigma_z = self.encoder(y)
        eps = torch.randn_like(mu_z, **self.tensor_params)
        z = mu_z + sigma_z * eps

        y_pred = self.decoder(z)
        decoder_loss = self.decoder.loss(y_obs=y,
                                         y_pred=y_pred,
                                         batch_scale=batch_scale,
                                         beta=beta)

        total_loss = decoder_loss + beta * batch_scale * self.KL_standard_normal(mu_z, sigma_z)

        return (z, mu_z, sigma_z), total_loss

    def optimize(self, data_loader: DataLoader, n_epochs: int, beta: float=1.0) -> None:
        '''
        Trains model.

        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/VAE.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch DataLoader containing dataset.
        num_epochs : int
            Number of epochs to train.
        '''

        # set initial condition to mean of dataset
        self.decoder.set_x0_(data_loader.dataset.tensors[0].detach().mean(dim=0, keepdim=True))

        # sample size
        sample_size = len(data_loader.dataset)

        # scaling for loglikelihood terms
        batch_scale = sample_size / data_loader.batch_size

        # format specifiers for printing loss and training progress
        loss_fmt_specifier_len = 8
        epoch_fmt_specifier_len = len(str(n_epochs))

        for epoch in range(1, n_epochs + 1):
            total_loss = 0

            for (batch, ) in data_loader:
                _, loss_on_batch = self.forward(y=batch,
                                                batch_scale=batch_scale,
                                                beta=beta)

                self.optimizer.zero_grad()
                loss_on_batch.backward()
                self.optimizer.step()

                total_loss += loss_on_batch.item()

            loss_fmt_specifier_len = self.__update_loss_fmt_specifier(total_loss, loss_fmt_specifier_len)
            print(f'[{int(epoch * 100 / n_epochs):3}%]\tEpoch: {epoch:{epoch_fmt_specifier_len}}/{n_epochs:{epoch_fmt_specifier_len}}\tTotal loss: {total_loss:{loss_fmt_specifier_len},.0f}',
                  end='\r', flush=True)
        print('\n') # needed so previous line isn't overwritten due to '\r' at end of previous print()

    def KL_standard_normal(self, mu: torch.Tensor, sigma: torch.Tensor) -> float:
        '''
        Adapted from https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/helpers.py
        Original author: Kaspar Märtens

        Parameters
        ----------
        mu : _type_
            _description_
        sigma : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        p = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        q = torch.distributions.normal.Normal(mu, sigma)

        return torch.sum(torch.distributions.kl_divergence(q, p))

    def __update_loss_fmt_specifier(self, total_loss: float, loss_fmt_specifier_len: int) -> int:
        '''
        Returns updated loss format specifier. 
        Helps keep the output pretty.

        Parameters
        ----------
        total_loss : float
            Current model loss
        loss_fmt_specifier : int
            Format specifier for previous loss.

        Returns
        -------
        int
            Updated loss format specifier
        '''
        if loss_fmt_specifier_len < (loss_str_len := len(f'{total_loss:,.0f}') ):
            return loss_str_len
        return loss_fmt_specifier_len

def build_VAE(params: dict) -> VAE:
    '''
    Builds BasisDeVAE model.

    Parameters
    ----------
    params : dict
        Parameters to pass to the Encoder, 
        Decoder, and BasisDeVAE

    Returns
    -------
    BasisDeVAE
        (torch.nn.Module)
    '''

    encoder_ = Encoder(**params['encoder'])
    decoder_ = BasisODEDecoder(**params['decoder'])

    return VAE(encoder=encoder_,
                      decoder=decoder_,
                      **params['vae'])

def train_model(model: VAE, data_loader: DataLoader, ds_name: str, epochs: int, save: bool=True, fname_base: str=None, save_dir: str='./') -> None:
    '''
    Wrapper for the torch.Module.optimize function.

    Parameters
    ----------
    model : BasisDeVAE
        BasisDeVAE model.
    data_loader : DataLoader
        Train dataset.
    ds_name : str
        Name of dataset.
    n_epochs : int
        Number of training epochs.
    logging_freq : int
        Frequency, in epochs, to print training metrics.
    save : bool, optional
        Whether to save the model's weights, by default `True`.
    fname_base : str, optional
        Base for output file name if `save` is `True`, by default 
        `None`.
    save_dir : str, optional
        Output directory if `save` is `True`, by default "./"
    '''

    model.optimize(data_loader, n_epochs=epochs)

    if not exists(save_dir):
        makedirs(save_dir, exist_ok=True)

    if save:
        if fname_base == None:
            fname_base = 'BasisDeVAE'
        torch.save(
            model.state_dict(),
            f'{save_dir}/{fname_base}_{ds_name}.pt',
        )

def get_cluster_assignments(model: VAE) -> np.ndarray:
    '''
    Gets clustering assignments from `model`.

    Adapted from https://github.com/djdanks/BasisDeVAE/blob/main/main.py
    Original author: Dominic Danks

    Parameters
    ----------
    model : BasisDeVAE
        BasisDeVAE model

    Returns
    -------
    np.ndarray
        Clustering assignments
    '''
    _model = model.eval()
    with torch.no_grad():
        cluster_probs = _model.decoder.get_phi().cpu().numpy()
        return cluster_probs.argmax(axis=1)
