import itertools as it
from collections import OrderedDict

import numpy as np
from tables import Hierarchical3GratingConfig, Hierarchical3GratingResult

from utils.utils import make_hash

configs = OrderedDict(
    seed=[42],
    n_draws_per_chain=[300],
    n_chains=[5],
    n_cores=[4],
    n_burnin=[200],
    gratings_offset=list(np.linspace(0, 0.8, 5)),  # by hand
    g_prob=[1 / 3],
    x_sigma=[0.1, 0.3, 0.8, 1],
    i_sigma=[0.1, 0.3, 0.8, 1],
    neuron_idx=[0],
)

config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    config["config_id"] = make_hash(config)
    config_list.append(config)

Hierarchical3GratingConfig.insert(config_list, skip_duplicates=True)
# CenterExperimentResult.populate()
Hierarchical3GratingResult.populate(reserve_jobs=True, order="random")
