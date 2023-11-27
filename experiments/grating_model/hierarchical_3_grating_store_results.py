import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.dj_experiments.tables import (
    Hierarchical3GratingConfig,
    Hierarchical3GratingResult,
)

results = Hierarchical3GratingConfig() * Hierarchical3GratingResult()

results_dict = {
    "gratings_offset": [],
    "x_sigma": [],
    "i_sigma": [],
    "mei_means": [],
    "mei_mean_mean_chains": [],
    "mei_mean_sde_chains": [],
    "completing_means": [],
    "completing_mean_mean_chains": [],
    "completing_mean_sde_chains": [],
    "disrupting_means_1": [],
    "disrupting_mean_mean_chains_1": [],
    "disrupting_mean_sde_chains_1": [],
    "disrupting_means_2": [],
    "disrupting_mean_mean_chains_2": [],
    "disrupting_mean_sde_chains_2": [],
    "completing_gain": [],
    "completing_gain_mean_chains": [],
    "completing_gain_sde_chains": [],
    "disrupting_gain_1": [],
    "disrupting_gain_mean_chains_1": [],
    "disrupting_gain_sde_chains_1": [],
    "disrupting_gain_2": [],
    "disrupting_gain_mean_chains_2": [],
    "disrupting_gain_sde_chains_2": [],
}
for idx, row in enumerate(results):
    print(f"Processing {idx+1}/{len(results)}")
    # if idx > 5:
    #     break
    results_dict["gratings_offset"].append(row["gratings_offset"])
    results_dict["x_sigma"].append(row["x_sigma"])
    results_dict["i_sigma"].append(row["i_sigma"])

    with Path(row["posterior_samples"]).open("rb") as f:
        (
            mei_samples,
            completing_samples,
            disrupting_samples_1,
            disrupting_samples_2,
        ) = pickle.load(f)

    n_chains = mei_samples["posterior"]["X"].data.shape[0]
    neuron_id = 12

    mei_mean = mei_samples["posterior"]["X"].data[..., neuron_id].mean(axis=1)
    mei_mean_mean_chains = (
        mei_samples["posterior"]["X"].data[..., neuron_id].mean(axis=1).mean(axis=0)
    )
    mei_mean_sde_chains = mei_samples["posterior"]["X"].data[..., neuron_id].mean(
        axis=1
    ).std(axis=0) / np.sqrt(n_chains)
    results_dict["mei_means"].append(mei_mean)
    results_dict["mei_mean_mean_chains"].append(mei_mean_mean_chains)
    results_dict["mei_mean_sde_chains"].append(mei_mean_sde_chains)

    completing_mean = (
        completing_samples["posterior"]["X"].data[..., neuron_id].mean(axis=1)
    )
    completing_mean_mean_chains = (
        completing_samples["posterior"]["X"]
        .data[..., neuron_id]
        .mean(axis=1)
        .mean(axis=0)
    )
    completing_mean_sde_chains = completing_samples["posterior"]["X"].data[
        ..., neuron_id
    ].mean(axis=1).std(axis=0) / np.sqrt(n_chains)
    results_dict["completing_means"].append(completing_mean)
    results_dict["completing_mean_mean_chains"].append(completing_mean_mean_chains)
    results_dict["completing_mean_sde_chains"].append(completing_mean_sde_chains)

    disrupting_mean_1 = (
        disrupting_samples_1["posterior"]["X"].data[..., neuron_id].mean(axis=1)
    )
    disrupting_mean_mean_chains_1 = (
        disrupting_samples_1["posterior"]["X"]
        .data[..., neuron_id]
        .mean(axis=1)
        .mean(axis=0)
    )
    disrupting_mean_sde_chains_1 = disrupting_samples_1["posterior"]["X"].data[
        ..., neuron_id
    ].mean(axis=1).std(axis=0) / np.sqrt(n_chains)
    results_dict["disrupting_means_1"].append(disrupting_mean_1)
    results_dict["disrupting_mean_mean_chains_1"].append(disrupting_mean_mean_chains_1)
    results_dict["disrupting_mean_sde_chains_1"].append(disrupting_mean_sde_chains_1)

    disrupting_mean_2 = (
        disrupting_samples_2["posterior"]["X"].data[..., neuron_id].mean(axis=1)
    )
    disrupting_mean_mean_chains_2 = (
        disrupting_samples_2["posterior"]["X"]
        .data[..., neuron_id]
        .mean(axis=1)
        .mean(axis=0)
    )
    disrupting_mean_sde_chains_2 = disrupting_samples_2["posterior"]["X"].data[
        ..., neuron_id
    ].mean(axis=1).std(axis=0) / np.sqrt(n_chains)
    results_dict["disrupting_means_2"].append(disrupting_mean_2)
    results_dict["disrupting_mean_mean_chains_2"].append(disrupting_mean_mean_chains_2)
    results_dict["disrupting_mean_sde_chains_2"].append(disrupting_mean_sde_chains_2)

    completing_gain = (completing_mean - mei_mean) / mei_mean
    completing_gain_mean_chains = completing_gain.mean(axis=0)
    completing_gain_sde_chains = completing_gain.std(axis=0) / np.sqrt(n_chains)
    results_dict["completing_gain"].append(completing_gain)
    results_dict["completing_gain_mean_chains"].append(completing_gain_mean_chains)
    results_dict["completing_gain_sde_chains"].append(completing_gain_sde_chains)

    disrupting_gain_1 = (disrupting_mean_1 - mei_mean) / mei_mean
    disrupting_gain_mean_chains_1 = disrupting_gain_1.mean(axis=0)
    disrupting_gain_sde_chains_1 = disrupting_gain_1.std(axis=0) / np.sqrt(n_chains)
    results_dict["disrupting_gain_1"].append(disrupting_gain_1)
    results_dict["disrupting_gain_mean_chains_1"].append(disrupting_gain_mean_chains_1)
    results_dict["disrupting_gain_sde_chains_1"].append(disrupting_gain_sde_chains_1)

    disrupting_gain_2 = (disrupting_mean_2 - mei_mean) / mei_mean
    disrupting_gain_mean_chains_2 = disrupting_gain_2.mean(axis=0)
    disrupting_gain_sde_chains_2 = disrupting_gain_2.std(axis=0) / np.sqrt(n_chains)
    results_dict["disrupting_gain_2"].append(disrupting_gain_2)
    results_dict["disrupting_gain_mean_chains_2"].append(disrupting_gain_mean_chains_2)
    results_dict["disrupting_gain_sde_chains_2"].append(disrupting_gain_sde_chains_2)


df = pd.DataFrame(results_dict)
# save the df to csv
df.to_csv("hierarchical_3_grating_analysis.csv")
