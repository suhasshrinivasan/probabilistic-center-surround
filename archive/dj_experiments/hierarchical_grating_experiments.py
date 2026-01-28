from pathlib import Path

import numpy as np
from insilico_stimuli.stimuli import CenterSurround

from models.hierarchical_grating_model import HierarchicalGratingModel


def get_3_gratings():
    canvas_size = [36, 36]
    locations = [[18, 18]]  # center position
    sizes_total = [36]  # total size (center + surround)
    sizes_center = [0.5]  # portion of radius used for center circle
    sizes_surround = [0.5]  # defines the starting portion of radius for surround
    contrasts_center = [1.0]  # try 2 center contrasts
    contrasts_surround = [1.0]  # surround contrast
    orientations_center = [0.0, np.pi / 2, np.pi / 4]  # variable orientation
    orientations_surround = [0.0, np.pi / 2, np.pi / 4]  # center only
    spatial_frequencies_center = [0.2]  # fixed spatial frequency
    phases_center = [np.pi]  # center phases
    grey_levels = [0.0]  # fixed grey level
    # spatial_frequencies_surround = [0.1, 0.3]  # optional parameter, default: same as spatial_frequencies_center
    # phases_surround = [np.pi/4]                # optional parameter, default: same as phases_center

    center_surround = CenterSurround(
        canvas_size=canvas_size,
        locations=locations,
        sizes_total=sizes_total,
        sizes_center=sizes_center,
        sizes_surround=sizes_surround,
        contrasts_center=contrasts_center,
        contrasts_surround=contrasts_surround,
        orientations_center=orientations_center,
        orientations_surround=orientations_surround,
        spatial_frequencies_center=spatial_frequencies_center,
        phases_center=phases_center,
        grey_levels=grey_levels,
    )

    gratings = np.array(
        [
            center_surround.images()[idx]
            for idx in range(len(center_surround.images()))
            if center_surround.params_dict_from_idx(idx)["orientation_center"]
            == center_surround.params_dict_from_idx(idx)["orientation_surround"]
        ]
    )
    return gratings


def hierarchical_3_grating_experiment(
    config_id,
    seed,
    n_draws_per_chain,
    n_chains,
    n_cores,
    n_burnin,
    gratings_offset,
    g_prob,
    x_sigma,
    i_sigma,
    neuron_idx,
):
    print("Running experiment with config:")
    print(
        f"seed={seed},"
        f" n_draws_per_chain={n_draws_per_chain},"
        f" n_chains={n_chains},"
        f" cores={n_cores},"
        f" n_burnin={n_burnin},"
        f" gratings_offset={gratings_offset},"
        f" g_prob={g_prob},"
        f" x_sigma={x_sigma},"
        f" i_sigma={i_sigma},"
        f" neuron_idx={neuron_idx}"
    )

    gratings = get_3_gratings()

    model = HierarchicalGratingModel(
        gratings=gratings,
        G_prob=g_prob,
        X_sigma=x_sigma,
        I_sigma=i_sigma,
        gratings_offset=gratings_offset,
    )

    MEI_center = model.grating_crops[4][0].reshape(12, 12)
    empty_image = np.zeros((36, 36))
    empty_image[12:24, 12:24] = MEI_center
    MEI = empty_image

    completing_image = model.gratings[0]
    disrupting_image_1 = model.gratings[1]
    disrupting_image_1[12:24, 12:24] = MEI_center

    disrupting_image_2 = model.gratings[2]
    disrupting_image_2[12:24, 12:24] = MEI_center

    images = [MEI, completing_image, disrupting_image_1, disrupting_image_2]

    post_samples_list = []
    for image in images:
        post_samples = model.sample_posterior(
            image=image,
            n_samples=n_draws_per_chain,
            random_seed=seed,
            tune=n_burnin,
            chains=n_chains,
            cores=n_cores,
        )
        post_samples_list.append(post_samples)

    return post_samples_list


def get_4_gratings():
    canvas_size = [36, 36]
    locations = [[18, 18]]  # center position
    sizes_total = [36]  # total size (center + surround)
    sizes_center = [0.5]  # portion of radius used for center circle
    sizes_surround = [0.5]  # defines the starting portion of radius for surround
    contrasts_center = [1.0]  # try 2 center contrasts
    contrasts_surround = [1.0]  # surround contrast
    orientations_center = [
        0.0,
        np.pi / 2,
        np.pi / 4,
        -np.pi / 4,
    ]  # variable orientation
    orientations_surround = [0.0, np.pi / 2, np.pi / 4, -np.pi / 4]  # center only
    spatial_frequencies_center = [0.2]  # fixed spatial frequency
    phases_center = [np.pi]  # center phases
    grey_levels = [0.0]  # fixed grey level
    # spatial_frequencies_surround = [0.1, 0.3]  # optional parameter, default: same as spatial_frequencies_center
    # phases_surround = [np.pi/4]                # optional parameter, default: same as phases_center

    center_surround = CenterSurround(
        canvas_size=canvas_size,
        locations=locations,
        sizes_total=sizes_total,
        sizes_center=sizes_center,
        sizes_surround=sizes_surround,
        contrasts_center=contrasts_center,
        contrasts_surround=contrasts_surround,
        orientations_center=orientations_center,
        orientations_surround=orientations_surround,
        spatial_frequencies_center=spatial_frequencies_center,
        phases_center=phases_center,
        grey_levels=grey_levels,
    )

    gratings = np.array(
        [
            center_surround.images()[idx]
            for idx in range(len(center_surround.images()))
            if center_surround.params_dict_from_idx(idx)["orientation_center"]
            == center_surround.params_dict_from_idx(idx)["orientation_surround"]
        ]
    )
    return gratings


def create_4_grating_stimuli(model):
    MEI_center = model.grating_crops[4][0].reshape(12, 12)
    empty_image = np.zeros((36, 36))
    empty_image[12:24, 12:24] = MEI_center
    MEI = empty_image

    completing_image = model.gratings[0]
    disrupting_image_1 = model.gratings[1]
    disrupting_image_1[12:24, 12:24] = MEI_center

    disrupting_image_2 = model.gratings[2]
    disrupting_image_2[12:24, 12:24] = MEI_center

    disrupting_image_3 = model.gratings[3]
    disrupting_image_3[12:24, 12:24] = MEI_center

    return {
        "mei": MEI,
        "completing_image": completing_image,
        "disrupting_image_1": disrupting_image_1,
        "disrupting_image_2": disrupting_image_2,
        "disrupting_image_3": disrupting_image_3,
    }


def hierarchical_4_grating_experiment(
    config_id,
    seed,
    n_draws_per_chain,
    n_chains,
    n_cores,
    n_burnin,
    gratings_offset,
    g_prob,
    x_sigma,
    i_sigma,
):
    print("Running experiment with config:")
    print(
        f"seed={seed},"
        f" n_draws_per_chain={n_draws_per_chain},"
        f" n_chains={n_chains},"
        f" cores={n_cores},"
        f" n_burnin={n_burnin},"
        f" gratings_offset={gratings_offset},"
        f" g_prob={g_prob},"
        f" x_sigma={x_sigma},"
        f" i_sigma={i_sigma}"
    )

    gratings = get_4_gratings()

    model = HierarchicalGratingModel(
        gratings=gratings,
        G_prob=g_prob,
        X_sigma=x_sigma,
        I_sigma=i_sigma,
        gratings_offset=gratings_offset,
    )

    images_dict = create_4_grating_stimuli(model)

    samples_dict = {}
    neuron_id = 16
    for key, image in images_dict.items():
        post_samples = model.sample_posterior(
            image=image,
            n_samples=n_draws_per_chain,
            random_seed=seed,
            tune=n_burnin,
            chains=n_chains,
            cores=n_cores,
        )
        samples_dict[key + "_x_samples"] = post_samples["posterior"]["X"].data
        samples_dict[key + "_g_samples"] = post_samples["posterior"]["G"].data

        samples_dict[key + "_x_neuron_mean"] = (
            post_samples["posterior"]["X"]
            .data[..., neuron_id]
            .mean(axis=1)
            .mean(axis=0)
        )
        samples_dict[key + "_x_neuron_sde"] = post_samples["posterior"]["X"].data[
            ..., neuron_id
        ].mean(axis=1).std(axis=0) / np.sqrt(n_chains)

    for key in images_dict.keys():
        if key == "mei":
            continue
        samples_dict[key + "_x_neuron_gain"] = (
            (
                samples_dict[key + "_x_samples"][..., neuron_id]
                - samples_dict["mei_x_samples"][..., neuron_id]
            )
            .mean(axis=1)
            .mean(axis=0)
        )
        samples_dict[key + "_x_neuron_gain_sde"] = (
            samples_dict[key + "_x_samples"][..., neuron_id]
            - samples_dict["mei_x_samples"][..., neuron_id]
        ).mean(axis=1).std(axis=0) / np.sqrt(n_chains)

    return samples_dict
