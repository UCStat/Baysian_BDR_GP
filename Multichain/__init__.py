"""Multichain Module for GP Bayesian Framework."""

from .multichain_sampler_D1 import MultiChainSampler as MultiChainSampler_D1
from .multichain_sampler_Dgeneral import MultiChainSampler as MultiChainSampler_Dgeneral

__all__ = [
    'MultiChainSampler_D1',
    'MultiChainSampler_Dgeneral'
]

