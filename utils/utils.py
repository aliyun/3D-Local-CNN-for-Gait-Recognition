#! /usr/bin/env python
import pdb
import torch
import random
import numpy as np

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def set_seed(seed):
    """Seed the PRNG for the CPU, Cuda, numpy and Python"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_seed(seed):
    if seed == -1:
        seed = np.random.randint(1, 100000)
    set_seed(seed)
    # torch.backends.cudnn.enabled = False

    # save gpu memory
    torch.backends.cudnn.deterministic = True

    # for fixed size input, about 10% speed if True
    torch.backends.cudnn.benchmark = False

    return seed
