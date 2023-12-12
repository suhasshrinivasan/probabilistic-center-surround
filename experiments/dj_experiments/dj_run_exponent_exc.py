import itertools as it
from collections import OrderedDict

from probcs.datajoint.exc_tables import ExcExponentConfig, ExcExponentResult

from probcs.utils.utils import make_hash


def is_valid_g_prob(config):
    return config["g_prob"] >= 1 / config["g_dim"] / 5


def is_valid_config(config):
    return is_valid_g_prob(config)


configs = OrderedDict(
    seed=[42],
    g_dim=[5],
    g_prob=[1 / 10, 1 / 13, 1 / 15, 1 / 20],
    x_sigma=[0.3, 0.5, 0.7],
    i_sigma=[10, 15, 20, 25, 30],
    patterns_offset=[0],
    g_x_exponent=[1, 2, 3, 4],
    n_tune=[500],
    n_draws=[500],
    n_chains=[4],
    n_cores=[4],
)

# test config for debugging
# configs = OrderedDict(
#     seed=[42],
#     g_dim=[3],
#     g_prob=[1 / 6],
#     x_sigma=[0.5],
#     i_sigma=[20],
#     patterns_offset=[0],
#     g_x_exponent=[4],
#     n_tune=[5],
#     n_draws=[5],
#     n_chains=[2],
#     n_cores=[2],
# )

config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    if not is_valid_config(config):
        continue
    config["config_id"] = make_hash(config)
    config_list.append(config)

ExcExponentConfig.insert(config_list, skip_duplicates=True)
ExcExponentResult.populate(reserve_jobs=True, order="random")
