import itertools as it
from collections import OrderedDict

from probcs.datajoint.exc_tables import ExcConfig, ExcResult2

from probcs.utils.utils import make_hash

configs = OrderedDict(
    seed=[42],
    g_dim=[10, 20],
    g_prob=[1 / 10, 1 / 5],
    x_sigma=[0.1],
    i_sigma=[20, 10, 5],
    patterns_offset=[0],
    n_tune=[500],
    n_draws=[500],
    n_chains=[4],
    n_cores=[4],
)

# # test config for debugging
# configs = OrderedDict(
#     seed=[42],
#     g_dim=[2],
#     g_prob=[1 / 2],
#     x_sigma=[0.1],
#     i_sigma=[1],
#     patterns_offset=[0],
#     n_tune=[5],
#     n_draws=[5],
#     n_chains=[2],
#     n_cores=[2],
# )


config_list = []
for values in it.product(*configs.values()):
    config = {key: value for key, value in zip(configs.keys(), values)}
    config["config_id"] = make_hash(config)
    config_list.append(config)

ExcConfig.insert(config_list, skip_duplicates=True)
# CenterExperimentResult.populate()
ExcResult2.populate(reserve_jobs=True, order="random")
