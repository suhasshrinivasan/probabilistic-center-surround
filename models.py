import pymc as pm
import torch
from numpy.random import default_rng

from utils import rate_code

seed = 42
rng = default_rng(seed=seed)
torch.manual_seed(seed=seed)
EPSILON = 1e-12


class HierarchicalModel:
    def __init__(
        self,
        n_neurons,
        global_orientation_bounds,
        gabor_filters,
        orientation_preferences,
        stimulus_std,
        rate_code_signature="base",
        vonmises_loc=0,
        vonmises_kappa=1,
        baseline_firing_rate=1,
    ):
        self.n_neurons = n_neurons
        self.global_orientation_bounds = global_orientation_bounds
        self.vonmises_loc = vonmises_loc
        self.vonmises_kappa = vonmises_kappa
        self.orientation_preferences = orientation_preferences
        self.baseline_firing_rate = baseline_firing_rate
        self.gabor_filters = gabor_filters
        self.stimulus_std = stimulus_std
        self.rate_code_signature = rate_code_signature
        self.model = self.build_model()

    def build_model(self):
        model = pm.Model()
        with model:
            stimulus_data = pm.MutableData("stimulus_data", None)
            global_orientation = pm.Uniform(
                "global_orientation", *self.global_orientation_bounds
            )
            neurons = pm.Exponential(
                "neurons",
                lam=rate_code(
                    self.baseline_firing_rate,
                    global_orientation,
                    self.orientation_preferences,
                    self.vonmises_loc,
                    self.vonmises_kappa,
                    self.rate_code_signature,
                ),
                shape=self.n_neurons,
            )
            stimulus = pm.Normal(
                "stimulus",
                mu=self.gabor_filters.transpose(1, 2, 0) @ neurons,
                sigma=self.stimulus_std,
                observed=stimulus_data,
            )
        return model

    def sample_prior(self, n_samples):
        with self.model:
            idata = pm.sample_prior_predictive(n_samples, random_seed=42)
        return idata

    def sample_posterior(
        self,
        observed_stimulus,
        draws=1000,
        tunes=1000,
        chains=4,
        cores=1,
        random_seed=42,
        return_inferencedata=True,
    ):
        with self.model:
            pm.set_data({"stimulus_data": observed_stimulus})
            idata = pm.sample(
                draws=draws,
                tune=tunes,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=return_inferencedata,
            )
        return idata
