
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from probcs.models.pattern_completion_model import PatternCompletionModel, BinaryPatternCompletionModel
from probcs.experiment_running.exc_experiment import center_surround_experiment, create_stimuli
from probcs.utils.utils import load_images, get_center_patches
from torchvision import transforms
import arviz as az
from sklearn.metrics.pairwise import cosine_similarity
from probcs.utils.utils import select_focus_images, get_excitatory_images
from probcs.utils.plotting import plot_posterior_binary
import pickle
import sys
import os

# Add the parent directory to the path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import binary_exc_model


def setup_experiment():
    """Set up the experiment configuration and model."""
    config = binary_exc_model['config_fn']()
    model = binary_exc_model['model_fn'](config)
    return config, model


def load_and_normalize_gratings(gratings_fname, exc_images):
    """Load gratings from file and normalize them based on excitatory images statistics."""
    gratings = np.load(gratings_fname)
    gratings_normed = gratings * exc_images.std() / gratings.std()
    return gratings_normed


def process_single_stimulus(model, stimulus, config, output_dir="gratings_results", 
                          results_prefix="results", stimulus_prefix="presented"):
    """Process a single stimulus through the model and save results."""
    idata = model(
        image=stimulus,
        n_samples=config['n_draws'],
        tune=config['n_tune'],
        chains=config['n_chains'],
        cores=config['n_cores'],
        random_seed=config['seed'],
    )
    return idata


def save_results(idata, stimulus, idx, output_dir="gratings_results", 
                results_prefix="results", stimulus_prefix="presented"):
    """Save model results and stimulus to files."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save results
    results_filename = f"{output_dir}/{results_prefix}_{idx}.pkl"
    with open(results_filename, "wb") as f:
        pickle.dump(idata, f)
    
    # Save stimulus
    stimulus_filename = f"{output_dir}/{stimulus_prefix}_{idx}.npy"
    np.save(stimulus_filename, stimulus)


def process_stimulus_batch(stimuli, model, config, experiment_name, stimulus_type=""):
    """Process a batch of stimuli through the model."""
    print(f"Processing {len(stimuli)} {stimulus_type} stimuli for {experiment_name}...")
    
    for idx, stimulus in enumerate(stimuli):
        print(f"Processing {stimulus_type} stimulus {idx + 1}/{len(stimuli)}")
        
        # Process stimulus
        idata = process_single_stimulus(model, stimulus, config)
        
        # Save results
        results_prefix = f"{experiment_name}_{stimulus_type}_results" if stimulus_type else f"{experiment_name}_results"
        stimulus_prefix = f"{experiment_name}_{stimulus_type}_presented" if stimulus_type else f"{experiment_name}_presented"
        
        save_results(idata, stimulus, idx, 
                    results_prefix=results_prefix,
                    stimulus_prefix=stimulus_prefix)


def run_gratings_experiment():
    """Run the complete gratings experiment."""
    # Setup
    config, model = setup_experiment()
    
    # Load and normalize gratings
    gratings_fname = "/src/project/experiments/exc_driven_model/coen_cagli_experiments/coen-cagli-gratings.npy"
    gratings_normed = load_and_normalize_gratings(gratings_fname, config['exc_images'])
    
    # Process full-size gratings
    process_stimulus_batch(
        gratings_normed, 
        model, 
        config, 
        experiment_name="coen_cagli",
        stimulus_type="big_gratings"
    )
    
    # Process center patches
    small_gratings = get_center_patches(gratings_normed)
    process_stimulus_batch(
        small_gratings, 
        model, 
        config, 
        experiment_name="coen_cagli",
        stimulus_type="small_gratings"
    )
    
    print("Gratings experiment completed!")


def main():
    """Main function to run the experiment."""
    run_gratings_experiment()


if __name__ == "__main__":
    main()