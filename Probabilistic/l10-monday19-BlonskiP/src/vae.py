from typing import Tuple

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from src.ae import BaseAutoEncoder


class VEncoder(nn.Module):
    """Encoder for VAE."""

    def __init__(
            self, n_input_features: int, n_hidden_neurons: int, n_latent_features: int,
    ):
        """ Implement Encoder neural network with given params.

        :param n_input_features: number of input features (28 x 28 = 784 for MNIST)
        :param n_hidden_neurons: number of neurons in hidden FC layer
        :param n_latent_features: size of the latent vector
        """
        super().__init__()
        self.n_input_features = n_input_features
        self.n_hidden_neurons = n_hidden_neurons
        self.n_latent_features = n_latent_features
        self.full_connected_layer_input = nn.Linear(self.n_input_features, self.n_hidden_neurons)
        self.full_connected_layer_hidden = nn.Linear(self.n_hidden_neurons, self.n_latent_features)
        self.full_connected_layer_laten = nn.Linear(self.n_hidden_neurons, self.n_latent_features)
        self.activation = nn.Softplus()  # activation function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implement encoding data to gaussian distribution params."""
        x = x.reshape(-1, self.n_input_features)

        first_layer_output = self.full_connected_layer_input(x)
        activation_first_layer = self.activation(first_layer_output)

        loc = self.full_connected_layer_hidden(activation_first_layer)
        scale = torch.exp(self.full_connected_layer_laten(activation_first_layer))

        return loc, scale


class VDecoder(nn.Module):
    """Decoder for VAE."""

    def __init__(
            self, n_latent_features: int, n_hidden_neurons: int, n_output_features: int,
    ):
        """ Implement Decoder neural network with given params.

        :param n_latent_features: number of latent features (same as in Encoder)
        :param n_hidden_neurons: number of neurons in hidden FC layer
        :param n_output_features: size of the output vector (28 x 28 = 784 for MNIST)
        """
        super().__init__()
        self.n_latent_features = n_latent_features
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_features = n_output_features

        self.full_connected_layer_laten_input = nn.Linear(self.n_latent_features, self.n_hidden_neurons)
        self.full_connected_layer_out = nn.Linear(self.n_hidden_neurons, self.n_output_features)
        self.activation = nn.Softmax()  # activation function

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Implement decoding latent vector to image."""
        hidden = self.full_connected_layer_laten_input(z)
        activate_hidden = self.softplus(hidden)

        loc_img = self.sigmoid(self.full_connected_layer_out(activate_hidden))
        return loc_img


class VariationalAutoEncoder(BaseAutoEncoder):
    """Variational Auto Encoder model."""

    def __init__(
            self,
            n_data_features: int,
            n_encoder_hidden_features: int,
            n_decoder_hidden_features: int,
            n_latent_features: int,
    ):
        """ Implement Variational Autoencoder with Pyro tools.

        :param n_data_features: number of input and output features (28 x 28 = 784 for MNIST)
        :param n_encoder_hidden_features: number of neurons in encoder's hidden layer
        :param n_decoder_hidden_features: number of neurons in decoder's hidden layer
        :param n_latent_features: number of latent features
        """
        encoder = VEncoder(
            n_input_features=n_data_features,
            n_hidden_neurons=n_encoder_hidden_features,
            n_latent_features=n_latent_features,
        )
        decoder = VDecoder(
            n_latent_features=n_latent_features,
            n_hidden_neurons=n_decoder_hidden_features,
            n_output_features=n_data_features,
        )
        super().__init__(
            encoder=encoder, decoder=decoder, n_latent_features=n_latent_features
        )

    def model(self, x: torch.Tensor):
        """Implement Pyro model for VAE; p(x|z)p(z)."""
        pyro.module("decoder", self.decoder)  # module register
        with pyro.plate("data", x.shape[0]):
            loc = x.new_zeros(torch.Size((x.shape[0], self.n_latent_features)))
            scale = x.new_ones(torch.Size((x.shape[0], self.n_latent_features)))
            # sampling laten_vector Z
            z = pyro.sample("latent", dist.Normal(loc, scale).to_event(1))
            # decode the latent vector z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    def guide(self, x: torch.Tensor):
        """Implement Pyro guide for VAE; q(z|x)"""
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Implement function to perform forward pass through encoder network.

        takes: tensor of shape [batch_size x input_flattened_size] (flattened input)
        returns: tensor of shape [batch_size x latent_feature_size] (latent vector)
        """
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            return pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        """ Implement unction to perform forward pass through decoder network.

        takes: tensor of shape [batch_size x latent_feature_size] (latent vector)
        returns: tensor of shape [batch_size x output_flattened_size] (flattened output)
        """
        with pyro.plate("data", z.shape[0]):
            loc_img = self.decoder.forward(z)
            return loc_img
