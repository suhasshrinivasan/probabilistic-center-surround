import os
import pickle
from pathlib import Path

import datajoint as dj
import numpy as np

from cs_experiments import center_experiment

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
class CenterExperimentConfig(dj.Manual):
    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    g_dim: int  # dimensionality of G
    n_draws_per_chain: int  # number of samples to draw per chain
    n_chains: int # number of chains
    cores: int  # number of cores to use
    n_burnin: int   # number of initial samples to discard
    contrast_scale: float   # scale to multiply with the stimulus
    x_sigma: float  # std of the noise in X
    i_sigma: float  # std of the noise in I
    stimulus_type: int # 0: -1 blank, nonnegative int represents the number of the latent whose RF would be the stimulus center
    """


@schema
class CenterExperimentResult(dj.Computed):
    definition = """
    -> CenterExperimentConfig
    ---
    posterior_samples: attach@external
    """

    def make(self, key):
        config = (CenterExperimentConfig & key).fetch1()
        posterior_samples = center_experiment(**config)
        filepath = Path(f"/tmp/{key['config_id']}.pkl")
        with filepath.open("wb") as f:
            pickle.dump(posterior_samples, f)
        key["posterior_samples"] = filepath
        self.insert1(key)
        filepath.unlink()
