from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import MLP


class ImpalaDecoder(nn.Module):
    """Decoder that mirrors ImpalaEncoder architecture for VAE reconstruction."""

    stack_sizes: tuple = (32, 32, 16)  # Reversed from encoder
    output_channels: int = 9  # frame_stack * 3 (RGB)
    mlp_hidden_dims: Sequence[int] = (512,)
    spatial_size: int = 8  # Size after encoder's conv layers (64 / 2^3 = 8)
    final_features: int = 32  # Last encoder stack size

    @nn.compact
    def __call__(self, z, train=True):
        # MLP to spatial features
        x = MLP(list(self.mlp_hidden_dims) + [self.spatial_size * self.spatial_size * self.final_features], activate_final=True)(z)
        x = x.reshape((*z.shape[:-1], self.spatial_size, self.spatial_size, self.final_features))

        initializer = nn.initializers.xavier_uniform()

        # Transposed conv stacks (reverse of encoder)
        for idx, num_features in enumerate(self.stack_sizes):
            # Upsample via transposed conv
            x = nn.ConvTranspose(
                features=num_features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                kernel_init=initializer,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                features=num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(x)
            x = nn.relu(x)

        # Final conv to output channels
        x = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=initializer,
        )(x)

        return x  # Output in [0, 1] range (sigmoid applied in loss)


class VAEModel(nn.Module):
    """VAE model with encoder and decoder for reconstruction pretraining."""

    encoder: nn.Module
    latent_dim: int = 512
    decoder_stack_sizes: tuple = (32, 32, 16)
    output_channels: int = 9  # frame_stack * 3

    @nn.compact
    def __call__(self, observations, train=True):
        # Encode to get features
        features = self.encoder(observations, train=train)

        # Project to mean and log variance
        mu = nn.Dense(self.latent_dim)(features)
        log_var = nn.Dense(self.latent_dim)(features)

        # Reparameterization trick
        if train:
            std = jnp.exp(0.5 * log_var)
            rng = self.make_rng('reparameterize')
            eps = jax.random.normal(rng, std.shape)
            z = mu + eps * std
        else:
            z = mu

        # Decode
        decoder = ImpalaDecoder(
            stack_sizes=self.decoder_stack_sizes,
            output_channels=self.output_channels,
        )
        reconstruction = decoder(z, train=train)

        return reconstruction, mu, log_var

    def encode(self, observations, train=True):
        """Encode observations to latent mean (for downstream use)."""
        features = self.encoder(observations, train=train)
        mu = nn.Dense(self.latent_dim)(features)
        return mu


class ATCModel(nn.Module):
    """ATC model with encoder, predictor, and bilinear similarity."""

    encoder: nn.Module
    predictor_hidden_dims: Sequence[int] = (512, 512)

    @nn.compact
    def __call__(self, observations, train=True):
        if hasattr(self.encoder, 'encode_with_info'):
            codes, vq_info = self.encoder.encode_with_info(observations, train=train)
        else:
            codes = self.encoder(observations, train=train)
            vq_info = None
        predictor_dims = list(self.predictor_hidden_dims) + [codes.shape[-1]]
        preds = MLP(predictor_dims, activate_final=False)(codes)
        preds = preds + codes
        bilinear = self.param(
            'bilinear',
            nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
            (codes.shape[-1], codes.shape[-1]),
        )
        return codes, preds, bilinear, vq_info

    def encode(self, observations, train=True):
        if hasattr(self.encoder, 'encode_with_info'):
            codes, _ = self.encoder.encode_with_info(observations, train=train)
            return codes
        return self.encoder(observations, train=train)
