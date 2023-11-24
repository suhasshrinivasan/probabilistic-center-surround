import itertools as it
from collections import OrderedDict

import numpy as np
from tables import Hierarchical4GratingConfig, Hierarchical4GratingResult

from utils.utils import make_hash

configs = OrderedDict(
    seed=[42],
    n_draws_per_chain=[300],
    n_chains=[5],
    n_cores=[4],
    n_burnin=[300],
    gratings_offset=list(np.linspace(0, 0.8, 5)),  # by hand
    g_prob=[1 / 3, 1 / 4, 1 / 8, 1 / 16],
    x_sigma=[0.1, 0.3, 0.8, 1],
    i_sigma=[0.1, 0.3, 0.8, 1],
)

# debug configs
# configs = OrderedDict(
#     seed=[42],
#     n_draws_per_chain=[10],
#     n_chains=[2],
#     n_cores=[2],
#     n_burnin=[1],
#     gratings_offset=list(np.linspace(0, 0.8, 5)),  # by hand
#     g_prob=[1 / 3, 1 / 4, 1 / 8, 1 / 16],
#     x_sigma=[0.1, 0.3, 0.8, 1],
#     i_sigma=[0.1, 0.3, 0.8, 1],
# )

config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    config["config_id"] = make_hash(config)
    config_list.append(config)

Hierarchical4GratingConfig.insert(config_list, skip_duplicates=True)
Hierarchical4GratingResult.populate(reserve_jobs=True, order="random")
