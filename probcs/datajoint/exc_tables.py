import os
import pickle
from pathlib import Path

import datajoint as dj

from ..experiment_running.exc_experiment import (
    exc_dj_experiment,
    exc_exponent_experiment,
    bernoulli_experiment,
    custom_binary_experiment,
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
class ExcConfig(dj.Manual):
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
class ExcResult(dj.Computed):
    definition = """
    -> ExcConfig
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
        config = (ExcConfig & key).fetch1()
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
        ) = exc_dj_experiment(**config)

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


@schema
class ExcResult2(dj.Computed):
    definition = """
    -> ExcConfig
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
        config = (ExcConfig & key).fetch1()
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
        ) = exc_dj_experiment(**config)

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


@schema
class ExcConfig10Plus(dj.Manual):
    """
    This table is for running experiments with g_dim >= 10
    """

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
class ExcResult10Plus(dj.Computed):
    """
    This table is for running experiments with g_dim >= 10
    """

    definition = """
    -> ExcConfig10Plus
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
        config = (ExcConfig10Plus & key).fetch1()
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
        ) = exc_dj_experiment(**config)

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


@schema
class ExcExponentConfig(dj.Manual):
    """
    Config table for experiments with G_X mapping with an exponent
    """

    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    g_dim: int  # dimensionality of G
    g_prob: float   # Bernoulli probability of g_i being 1
    x_sigma: float  # Laplace distribution scale parameter for x
    i_sigma: float  # Normal distribution scale parameter for i
    patterns_offset: float   # offset to increase cosine similarity between patterns
    g_x_exponent: int
    n_tune: int # number of tuning samples
    n_draws: int    # number of samples to draw
    n_chains: int  # number of chains
    n_cores: int  # number of cores to use
    """


@schema
class ExcExponentResult(dj.Computed):
    """
    Result table for experiments with G_X mapping with an exponent
    """

    definition = """
    -> ExcExponentConfig
    ---
    all_idata: attach@external
    all_stimuli: attach@external
    average_exc: float
    average_inh_1: float
    average_inh_2: float
    disrupting_pattern_indices: longblob
    """

    def make(self, key):
        config = (ExcExponentConfig & key).fetch1()
        (
            all_idata,
            all_stimuli,
            average_exc,
            average_inh_1,
            average_inh_2,
            disrupting_pattern_indices,
        ) = exc_exponent_experiment(**config)

        idata_filepath = Path(f"/tmp/{key['config_id']}_idata.pkl")
        with idata_filepath.open("wb") as f:
            pickle.dump(all_idata, f)
        key["all_idata"] = idata_filepath

        stimuli_filepath = Path(f"/tmp/{key['config_id']}_stimuli.pkl")
        with stimuli_filepath.open("wb") as f:
            pickle.dump(all_stimuli, f)
        key["all_stimuli"] = stimuli_filepath

        key["average_exc"] = average_exc
        key["average_inh_1"] = average_inh_1
        key["average_inh_2"] = average_inh_2
        key["disrupting_pattern_indices"] = disrupting_pattern_indices
        # key["fig_g_stimuli"] = fig_g_stimuli
        # key["fig_cossim_g"] = fig_cossim_g
        # key["list_fig_stimuli"] = list_fig_stimuli
        # key["fig_g_x_map"] = fig_g_x_map
        # key["fig_posterior"] = fig_posterior
        # key["fig_samples"] = fig_samples
        # fig_g_stimuli_filepath = Path(f"/tmp/{key['config_id']}_fig_g_stimuli.pkl")
        # with fig_g_stimuli_filepath.open("wb") as f:
        #     pickle.dump(fig_g_stimuli, f)

        # fig_cossim_g_filepath = Path(f"/tmp/{key['config_id']}_fig_cossim_g.pkl")
        # with fig_cossim_g_filepath.open("wb") as f:
        #     pickle.dump(fig_cossim_g, f)

        # list_fig_stimuli_filepath = Path(
        #     f"/tmp/{key['config_id']}_list_fig_stimuli.pkl"
        # )
        # with list_fig_stimuli_filepath.open("wb") as f:
        #     pickle.dump(list_fig_stimuli, f)

        # fig_g_x_map_filepath = Path(f"/tmp/{key['config_id']}_fig_g_x_map.pkl")
        # with fig_g_x_map_filepath.open("wb") as f:
        #     pickle.dump(fig_g_x_map, f)

        # fig_posterior_filepath = Path(f"/tmp/{key['config_id']}_fig_posterior.pkl")
        # with fig_posterior_filepath.open("wb") as f:
        #     pickle.dump(fig_posterior, f)

        # fig_samples_filepath = Path(f"/tmp/{key['config_id']}_fig_samples.pkl")
        # with fig_samples_filepath.open("wb") as f:
        #     pickle.dump(fig_samples, f)

        # key["fig_g_stimuli"] = fig_g_stimuli_filepath
        # key["fig_cossim_g"] = fig_cossim_g_filepath
        # key["list_fig_stimuli"] = list_fig_stimuli_filepath
        # key["fig_g_x_map"] = fig_g_x_map_filepath
        # key["fig_posterior"] = fig_posterior_filepath
        # key["fig_samples"] = fig_samples_filepath

        self.insert1(key)

        idata_filepath.unlink()
        stimuli_filepath.unlink()
        # fig_g_stimuli_filepath.unlink()
        # fig_cossim_g_filepath.unlink()
        # list_fig_stimuli_filepath.unlink()
        # fig_g_x_map_filepath.unlink()
        # fig_posterior_filepath.unlink()
        # fig_samples_filepath.unlink()


@schema
class ExcExponentCenterCropConfig(dj.Manual):
    """
    Config table for experiments with G_X mapping with an exponent with all images center cropped
    """

    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    g_dim: int  # dimensionality of G
    g_prob: float   # Bernoulli probability of g_i being 1
    x_sigma: float  # Laplace distribution scale parameter for x
    i_sigma: float  # Normal distribution scale parameter for i
    patterns_offset: float   # offset to increase cosine similarity between patterns
    g_x_exponent: int
    n_tune: int # number of tuning samples
    n_draws: int    # number of samples to draw
    n_chains: int  # number of chains
    n_cores: int  # number of cores to use
    """


@schema
class ExcExponentCenterCropResult(dj.Computed):
    """
    Result table for experiments with G_X mapping with an exponent with all images center cropped
    """

    definition = """
    -> ExcExponentCenterCropConfig
    ---
    all_idata: attach@external
    all_stimuli: attach@external
    average_exc: float
    average_inh_1: float
    average_inh_2: float
    disrupting_pattern_indices: longblob
    """

    def make(self, key):
        config = (ExcExponentCenterCropConfig & key).fetch1()
        pattern_fname = "/src/project/data/experiment/custom_centered_exc_crops.npy"
        (
            all_idata,
            all_stimuli,
            average_exc,
            average_inh_1,
            average_inh_2,
            disrupting_pattern_indices,
        ) = exc_exponent_experiment(
            exc_fname=pattern_fname,
            exc_image_ids=None,
            **config,
        )

        idata_filepath = Path(f"/tmp/{key['config_id']}_idata.pkl")
        with idata_filepath.open("wb") as f:
            pickle.dump(all_idata, f)
        key["all_idata"] = idata_filepath

        stimuli_filepath = Path(f"/tmp/{key['config_id']}_stimuli.pkl")
        with stimuli_filepath.open("wb") as f:
            pickle.dump(all_stimuli, f)
        key["all_stimuli"] = stimuli_filepath

        key["average_exc"] = average_exc
        key["average_inh_1"] = average_inh_1
        key["average_inh_2"] = average_inh_2
        key["disrupting_pattern_indices"] = disrupting_pattern_indices
        # key["fig_g_stimuli"] = fig_g_stimuli
        # key["fig_cossim_g"] = fig_cossim_g
        # key["list_fig_stimuli"] = list_fig_stimuli
        # key["fig_g_x_map"] = fig_g_x_map
        # key["fig_posterior"] = fig_posterior
        # key["fig_samples"] = fig_samples
        # fig_g_stimuli_filepath = Path(f"/tmp/{key['config_id']}_fig_g_stimuli.pkl")
        # with fig_g_stimuli_filepath.open("wb") as f:
        #     pickle.dump(fig_g_stimuli, f)

        # fig_cossim_g_filepath = Path(f"/tmp/{key['config_id']}_fig_cossim_g.pkl")
        # with fig_cossim_g_filepath.open("wb") as f:
        #     pickle.dump(fig_cossim_g, f)

        # list_fig_stimuli_filepath = Path(
        #     f"/tmp/{key['config_id']}_list_fig_stimuli.pkl"
        # )
        # with list_fig_stimuli_filepath.open("wb") as f:
        #     pickle.dump(list_fig_stimuli, f)

        # fig_g_x_map_filepath = Path(f"/tmp/{key['config_id']}_fig_g_x_map.pkl")
        # with fig_g_x_map_filepath.open("wb") as f:
        #     pickle.dump(fig_g_x_map, f)

        # fig_posterior_filepath = Path(f"/tmp/{key['config_id']}_fig_posterior.pkl")
        # with fig_posterior_filepath.open("wb") as f:
        #     pickle.dump(fig_posterior, f)

        # fig_samples_filepath = Path(f"/tmp/{key['config_id']}_fig_samples.pkl")
        # with fig_samples_filepath.open("wb") as f:
        #     pickle.dump(fig_samples, f)

        # key["fig_g_stimuli"] = fig_g_stimuli_filepath
        # key["fig_cossim_g"] = fig_cossim_g_filepath
        # key["list_fig_stimuli"] = list_fig_stimuli_filepath
        # key["fig_g_x_map"] = fig_g_x_map_filepath
        # key["fig_posterior"] = fig_posterior_filepath
        # key["fig_samples"] = fig_samples_filepath

        self.insert1(key)

        idata_filepath.unlink()
        stimuli_filepath.unlink()
        # fig_g_stimuli_filepath.unlink()
        # fig_cossim_g_filepath.unlink()
        # list_fig_stimuli_filepath.unlink()
        # fig_g_x_map_filepath.unlink()
        # fig_posterior_filepath.unlink()
        # fig_samples_filepath.unlink()


@schema
class CustomBinaryConfig(dj.Manual):
    """
    Config table for experiments with Bernoulli neurons but with custom g_x mapping
    """

    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    g_dim: int  # dimensionality of G
    g_prob: float   # Bernoulli probability of g_i being 1
    i_sigma: float  # Normal distribution scale parameter for i
    patterns_offset: float   # offset to increase cosine similarity between patterns
    image_type: varchar(30) # type of image to use
    n_tune: int # number of tuning samples
    n_draws: int    # number of samples to draw
    n_chains: int  # number of chains
    n_cores: int  # number of cores to use
    """


@schema
class CustomBinaryResult(dj.Computed):
    """
    Result table for experiments with Bernoulli neurons but with custom g_x mapping
    """

    definition = """
    -> CustomBinaryConfig
    ---
    all_idata: attach@external
    all_stimuli: attach@external
    disrupting_pattern_indices: longblob
    """

    def make(self, key):
        config = (CustomBinaryConfig & key).fetch1()
        (
            all_idata,
            all_stimuli,
            disrupting_pattern_indices,
        ) = custom_binary_experiment(
            **config,
        )

        idata_filepath = Path(f"/tmp/{key['config_id']}_idata.pkl")
        with idata_filepath.open("wb") as f:
            pickle.dump(all_idata, f)
        key["all_idata"] = idata_filepath

        stimuli_filepath = Path(f"/tmp/{key['config_id']}_stimuli.pkl")
        with stimuli_filepath.open("wb") as f:
            pickle.dump(all_stimuli, f)
        key["all_stimuli"] = stimuli_filepath

        key["disrupting_pattern_indices"] = disrupting_pattern_indices

        self.insert1(key)

        idata_filepath.unlink()
        stimuli_filepath.unlink()


@schema
class BernoulliConfig(dj.Manual):
    """
    Config table for experiments with Bernoulli neurons
    """

    definition = """
    config_id: char(32)
    ---
    seed: int   # random seed
    g_dim: int  # dimensionality of G
    g_prob: float   # Bernoulli probability of g_i being 1
    i_sigma: float  # Normal distribution scale parameter for i
    patterns_offset: float   # offset to increase cosine similarity between patterns
    offset_x_1: float  # offset to add to the mapping from G to X
    offset_x_2: float
    scale_x: float  # scale to multiply the mapping from G to X
    exponent_x: float # exponent to raise the mapping from G to X
    n_tune: int # number of tuning samples
    n_draws: int    # number of samples to draw
    n_chains: int  # number of chains
    n_cores: int  # number of cores to use
    """


@schema
class BernoulliResult(dj.Computed):
    """
    Result table for experiments with G_X mapping with an exponent with all images center cropped
    """

    definition = """
    -> BernoulliConfig
    ---
    all_idata: attach@external
    all_stimuli: attach@external
    average_exc: float
    average_inh_1: float
    average_inh_2: float
    disrupting_pattern_indices: longblob
    """

    def make(self, key):
        config = (BernoulliConfig & key).fetch1()
        (
            all_idata,
            all_stimuli,
            average_exc,
            average_inh_1,
            average_inh_2,
            disrupting_pattern_indices,
        ) = bernoulli_experiment(
            **config,
        )

        idata_filepath = Path(f"/tmp/{key['config_id']}_idata.pkl")
        with idata_filepath.open("wb") as f:
            pickle.dump(all_idata, f)
        key["all_idata"] = idata_filepath

        stimuli_filepath = Path(f"/tmp/{key['config_id']}_stimuli.pkl")
        with stimuli_filepath.open("wb") as f:
            pickle.dump(all_stimuli, f)
        key["all_stimuli"] = stimuli_filepath

        key["average_exc"] = average_exc
        key["average_inh_1"] = average_inh_1
        key["average_inh_2"] = average_inh_2
        key["disrupting_pattern_indices"] = disrupting_pattern_indices

        self.insert1(key)

        idata_filepath.unlink()
        stimuli_filepath.unlink()
