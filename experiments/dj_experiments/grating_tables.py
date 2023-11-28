import os
import pickle
from pathlib import Path

import datajoint as dj
import numpy as np
from insilico_stimuli.stimuli import CenterSurround, GaborSet

from experiments.exc_driven_model.exc_experiment import grating_dj_experiment

dj.config["enable_python_native_blobs"] = True

dj.config["stores"] = {
    "external": {
        "protocol": "s3",
        "endpoint": os.environ["MINIO_ENDPOINT"],
        "access_key": os.environ["MINIO_ACCESS_KEY"],
        "secret_key": os.environ["MINIO_SECRET_KEY"],
        "bucket": "neural-sampling-code",
        "location": "dj-store",
        "secure": True,
    }
}

schema = dj.schema("sshrinivasan_probcs")


@schema
class GratingConfig(dj.Manual):
    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    g_dim: int  # dimensionality of G
    g_prob: float   # Bernoulli probability of g_i being 1
    x_sigma: float  # Laplace distribution scale parameter for x
    i_sigma: float  # Normal distribution scale parameter for i
    patterns_offset: float   # offset to increase cosine similarity between patterns
    n_tune: int # number of tuning samples
    n_draws: int    # number of samples to draw
    n_chains: int  # number of chains
    n_cores: int  # number of cores to use
    """


@schema
class GratingResult(dj.Computed):
    definition = """
    -> GratingConfig
    ---
    all_idata: attach@external
    all_stimuli: attach@external
    all_g_means: longblob
    all_g_means_sde: longblob
    all_center_x_means: longblob
    all_center_x_means_sde: longblob 
    all_center_x_perc_change_means: longblob
    all_center_x_perc_change_means_sde: longblob
    """

    def make(self, key):
        config = (GratingConfig & key).fetch1()

        (
            _,
            all_idata,
            all_stimuli,
            all_g_means,
            all_g_means_sde,
            all_center_x_means,
            all_center_x_means_sde,
            all_center_x_perc_change_means,
            all_center_x_perc_change_means_sde,
        ) = grating_dj_experiment(**config)

        idata_filepath = Path(f"/tmp/{key['config_id']}_idata.pkl")
        with idata_filepath.open("wb") as f:
            pickle.dump(all_idata, f)
        key["all_idata"] = idata_filepath

        stimuli_filepath = Path(f"/tmp/{key['config_id']}_stimuli.pkl")
        with stimuli_filepath.open("wb") as f:
            pickle.dump(all_stimuli, f)
        key["all_stimuli"] = stimuli_filepath

        key["all_g_means"] = all_g_means
        key["all_g_means_sde"] = all_g_means_sde
        key["all_center_x_means"] = all_center_x_means
        key["all_center_x_means_sde"] = all_center_x_means_sde
        key["all_center_x_perc_change_means"] = all_center_x_perc_change_means
        key["all_center_x_perc_change_means_sde"] = all_center_x_perc_change_means_sde

        self.insert1(key)

        idata_filepath.unlink()
        stimuli_filepath.unlink()
