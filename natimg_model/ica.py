import argparse as ap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import FastICA
from utils import load_images

# global constants
FIG_DPI = 300


def perform_ica(images, n_components, negentropy, max_iter, whiten_solver, seed):
    # create ica model
    images = images.reshape(images.shape[0], -1)
    ica_model = FastICA(
        n_components=n_components,
        fun=negentropy,
        max_iter=max_iter,
        whiten=True,
        whiten_solver=whiten_solver,
        random_state=seed,
    )
    # fit ica model
    ica_model.fit(images)
    return ica_model


def visualize_filters(ica_model, n_components, savepath):
    # visualize the mixing matrix
    mixing_matrix = ica_model.mixing_
    image_length = int(np.sqrt(mixing_matrix.shape[0]))
    n_rows = int(np.sqrt(n_components))
    n_cols = int(np.sqrt(n_components))
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, dpi=FIG_DPI)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(mixing_matrix[:, i].reshape(image_length, image_length), cmap="gray")
        ax.axis("off")
    ax.set_aspect("equal")
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def visualize_component_density(
    ica_model, images, n_components, all_components_savepath, subset_components_savepath
):
    images = images.reshape(images.shape[0], -1)
    components = ica_model.transform(images)
    # visualize the component density
    nrows = int(np.sqrt(n_components))
    ncols = int(np.sqrt(n_components))
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, dpi=FIG_DPI, sharex=True, sharey=True
    )
    for component_values, ax in zip(components.T, axs.flatten()):
        sns.histplot(
            component_values,
            ax=ax,
            stat="probability",
            color="black",
        )
        ax.axis("off")
    fig.savefig(all_components_savepath, bbox_inches="tight")
    plt.close(fig)
    # visualize a smaller subset of the component density
    nrows = int(np.sqrt(n_components) / 2)
    ncols = int(np.sqrt(n_components) / 2)
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, dpi=FIG_DPI, sharex=True, sharey=True
    )
    for component_values, ax in zip(components.T, axs.flatten()):
        sns.histplot(
            component_values,
            ax=ax,
            stat="probability",
            color="black",
        )
    fig.savefig(subset_components_savepath, bbox_inches="tight")


def perform_repeat_ica(
    images,
    n_components,
    negentropy,
    max_iter,
    whiten_solver,
    n_repeats,
    meta_seed,
    save_path,
):
    # create seeds for each repeat
    rng = np.random.default_rng(meta_seed)
    seeds = rng.integers(low=0, high=2**32 - 1, size=n_repeats)
    # perform ica for each seed
    ica_models = []
    for seed in seeds:
        print(f"Performing ICA for seed {seed}...")
        # set savepaths for images, mixing matrices, component densities and models
        filters_savepath = save_path.joinpath(f"filters_{seed}.pdf")
        all_components_savepath = save_path.joinpath(f"all_components_{seed}.pdf")
        subset_components_savepath = save_path.joinpath(f"subset_components_{seed}.pdf")
        models_savepath = save_path.joinpath(f"model_{seed}.joblib")
        # perform ica
        ica_model = perform_ica(
            images, n_components, negentropy, max_iter, whiten_solver, seed
        )
        print(f"Visualizing ICA for seed {seed}...")
        # visualize the mixing matrix
        visualize_filters(ica_model, n_components, filters_savepath)
        # visualize the component density
        print(f"Visualizing component density for seed {seed}...")
        visualize_component_density(
            ica_model,
            images,
            n_components,
            all_components_savepath,
            subset_components_savepath,
        )
        # save ica model
        joblib.dump(ica_model, models_savepath)
        # append ica model to list
        ica_models.append(ica_model)
    return ica_models


def repeat_ica_handler(
    images_folder_basepath,
    root_crop_size,
    n_components,
    n_samples,
    negentropy,
    max_iter,
    whiten_solver,
    n_repeats,
    meta_seed,
    save_folder_basepath,
    root_n_images_save,
):
    # mkdir save folder
    save_path = Path(save_folder_basepath)
    save_path = save_path.joinpath(
        f"repeat_{n_repeats}_crop_{root_crop_size}_comp_{n_components}"
    )
    save_path.mkdir(parents=True, exist_ok=False)

    # create the savepaths for images, mixing matrices and models
    images_savepath = save_path.joinpath("images.pdf")

    print("Loading images...")
    # load images
    images = load_images(
        dataset_folder_path=images_folder_basepath,
        crop_size=root_crop_size,
        n_samples=n_samples,
    )
    # save sample of images as a fig
    fig, axs = plt.subplots(root_n_images_save, root_n_images_save, dpi=FIG_DPI)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    ax.set_aspect("equal")
    fig.savefig(f"{images_savepath}", bbox_inches="tight")
    plt.close(fig)

    # perform repeat ICA
    _ = perform_repeat_ica(
        images,
        n_components,
        negentropy,
        max_iter,
        whiten_solver,
        n_repeats,
        meta_seed,
        save_path,
    )


def main():
    parser = ap.ArgumentParser()

    print("Parsing arguments...")
    # image data arguments
    parser.add_argument(
        "--images_folder_basepath", type=str, default="/src/project/data/natimgs/"
    )
    parser.add_argument("--root_crop_size", type=int, default=12)

    # ICA arguments
    parser.add_argument("--n_components", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=20_000)
    parser.add_argument("--negentropy", type=str, default="logcosh")
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--whiten_solver", type=str, default="svd")

    # center-surround panelling arguments
    parser.add_argument("--n_repeats", type=int, default=9)
    parser.add_argument("--meta_seed", type=int, default=0)

    # save arguments
    parser.add_argument(
        "--save_folder_basepath", type=str, default="/src/project/data/ica/"
    )
    parser.add_argument("--root_n_images_save", type=int, default=20)

    # parse arguments
    args = vars(parser.parse_args())

    # perform arg checks
    assert (
        args["n_components"] <= args["root_crop_size"] ** 2
    ), "n_components must be <= image size"
    assert (
        args["n_samples"] >= 50 * args["n_components"]
        if args["whiten_solver"] == "eigh"
        else True
    ), "n_samples must be >= 50 * n_components if whiten_solver == 'eigh'"
    assert args["n_samples"] <= 20_000, "n_samples must be <= 24k"

    print("Calling repeat ICA handler...")
    repeat_ica_handler(**args)


if __name__ == "__main__":
    main()
