import itertools as it
from collections import OrderedDict

from probcs.datajoint.exc_tables import CustomBinaryConfig, CustomBinaryResult

from probcs.utils.utils import make_hash


def is_valid_g_prob(config):
    return config["g_prob"] >= 1 / config["g_dim"] / 5


def is_valid_config(config):
    return is_valid_g_prob(config)


configs = OrderedDict(
    seed=[42],
    g_dim=[5],
    g_prob=[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    i_sigma=[60, 80, 100, 120],
    patterns_offset=[0],
    image_type=["exc"],
    n_tune=[600],
    n_draws=[600],
    n_chains=[4],
    n_cores=[4],
)

# test config for debugging
# configs = OrderedDict(
#     seed=[42],
#     g_dim=[5],
#     g_prob=[0.1, 0.2, 0.4, 0.5, 0.6],
#     i_sigma=[60, 80, 100, 120],
#     patterns_offset=[0],
#     n_tune=[5],
#     n_draws=[5],
#     n_chains=[4],
#     n_cores=[4],
# )

config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    # if not is_valid_config(config):
    #     continue
    config["config_id"] = make_hash(config)
    config_list.append(config)

CustomBinaryConfig.insert(config_list, skip_duplicates=True)
CustomBinaryResult.populate(reserve_jobs=True, order="random")
