import os
import pickle
from pathlib import Path

import datajoint as dj
import numpy as np

from experiments.dj_experiments.cs_experiments import center_experiment
from experiments.dj_experiments.hierarchical_grating_experiments import (
    hierarchical_3_grating_experiment,
    hierarchical_4_grating_experiment,
)

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


@schema
class Hierarchical3GratingConfig(dj.Manual):
    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    n_draws_per_chain: int  # number of samples to draw per chain
    n_chains: int # number of chains
    n_cores: int  # number of cores to use
    n_burnin: int   # number of initial samples to discard

    gratings_offset: float   # gratings offset to increase cosine similarity between gratings on top
    
    g_prob: float   # multinoulli probability of G
    x_sigma: float  # std of the noise in X
    i_sigma: float  # std of the noise in I

    neuron_idx: int # index of the neuron to use for the experiment
    """


@schema
class Hierarchical3GratingResult(dj.Computed):
    definition = """
    -> Hierarchical3GratingConfig
    ---
    posterior_samples: attach@external
    """

    def make(self, key):
        config = (Hierarchical3GratingConfig & key).fetch1()
        posterior_samples = hierarchical_3_grating_experiment(**config)
        filepath = Path(f"/tmp/{key['config_id']}.pkl")
        with filepath.open("wb") as f:
            pickle.dump(posterior_samples, f)
        key["posterior_samples"] = filepath
        self.insert1(key)
        filepath.unlink()


@schema
class Hierarchical4GratingConfig(dj.Manual):
    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    n_draws_per_chain: int  # number of samples to draw per chain
    n_chains: int # number of chains
    n_cores: int  # number of cores to use
    n_burnin: int   # number of initial samples to discard

    gratings_offset: float   # gratings offset to increase cosine similarity between gratings on top
    
    g_prob: float   # multinoulli probability of G
    x_sigma: float  # std of the noise in X
    i_sigma: float  # std of the noise in I
    """


@schema
class Hierarchical4GratingResult(dj.Computed):
    definition = """
    -> Hierarchical4GratingConfig
    ---
    mei_x_samples : longblob
    mei_g_samples : longblob
    mei_x_neuron_mean : float
    mei_x_neuron_sde : float
    completing_image_x_samples : longblob
    completing_image_g_samples : longblob
    completing_image_x_neuron_mean : float
    completing_image_x_neuron_sde : float
    disrupting_image_1_x_samples : longblob
    disrupting_image_1_g_samples : longblob
    disrupting_image_1_x_neuron_mean : float
    disrupting_image_1_x_neuron_sde : float
    disrupting_image_2_x_samples : longblob
    disrupting_image_2_g_samples : longblob
    disrupting_image_2_x_neuron_mean : float
    disrupting_image_2_x_neuron_sde : float
    disrupting_image_3_x_samples : longblob
    disrupting_image_3_g_samples : longblob
    disrupting_image_3_x_neuron_mean : float
    disrupting_image_3_x_neuron_sde : float
    completing_image_x_neuron_gain : float
    completing_image_x_neuron_gain_sde : float
    disrupting_image_1_x_neuron_gain : float
    disrupting_image_1_x_neuron_gain_sde : float
    disrupting_image_2_x_neuron_gain : float
    disrupting_image_2_x_neuron_gain_sde : float
    disrupting_image_3_x_neuron_gain : float
    disrupting_image_3_x_neuron_gain_sde : float
    """

    def make(self, key):
        config = (Hierarchical4GratingConfig & key).fetch1()
        samples_dict = hierarchical_4_grating_experiment(**config)
        # filepath = Path(f"/tmp/{key['config_id']}.pkl")
        # with filepath.open("wb") as f:
        #     pickle.dump(samples_dict, f)
        # key["samples_dict"] = filepath
        # self.insert1(key)
        # filepath.unlink()
        samples_dict["config_id"] = key["config_id"]
        self.insert1(samples_dict)
