import itertools as it
from collections import OrderedDict

from utils.utils import make_hash

from tables import CenterExperimentConfig, CenterExperimentResult

latent_ids = range(160, 200, 1)

configs = OrderedDict(
    seed=[42],
    g_dim=[340],
    n_draws_per_chain=[1000],
    n_chains=[4],
    cores=[4],
    n_burnin=[1000],
    contrast_scale=[1.0],
    x_sigma=[0.1],
    i_sigma=[0.1],
    stimulus_type=[
        -1,
        *latent_ids,
    ],
)

config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    config["config_id"] = make_hash(config)
    config_list.append(config)

CenterExperimentConfig.insert(config_list, skip_duplicates=True)
# CenterExperimentResult.populate()
CenterExperimentResult.populate(reserve_jobs=True, order="random")
