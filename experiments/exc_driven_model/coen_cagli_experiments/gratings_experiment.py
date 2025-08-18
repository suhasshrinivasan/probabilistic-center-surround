# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from probcs.models.pattern_completion_model import PatternCompletionModel, BinaryPatternCompletionModel
from probcs.experiment_running.exc_experiment import center_surround_experiment, create_stimuli
from probcs.utils.utils import load_images
from torchvision import transforms
import arviz as az
from sklearn.metrics.pairwise import cosine_similarity
from probcs.utils.utils import select_focus_images, get_excitatory_images
from probcs.utils.plotting import plot_posterior_binary
import pickle
import pandas as pd
import os
import glob

seed = 42
rng = np.random.default_rng(seed)

# %%
results_dict = {
    'stimulus_id': [],
    'big_stimulus': [],
    'small_x_mean': [],
    'small_g_mean': [],
    'big_x_mean': [],
    'big_g_mean': [],
}

# Find all small stimulus .pkl files in the results directory
small_pkl_files = glob.glob("results/coen_cagli_small_stimulus_*.pkl")
print(f"Found {len(small_pkl_files)} small stimulus .pkl files:")
for file in small_pkl_files:
    print(f"  {file}")

# Process each small stimulus .pkl file
for results_path in small_pkl_files:
    # Load the small stimulus pickle file
    with open(results_path, "rb") as f:
        results = pickle.load(f)
        X = results['posterior']['X'].data
        X_mean = X.mean((0, 1))
        G = results['posterior']['G'].data
        G_mean = G.mean((0, 1))
        
        # Extract ID from filename
        stimulus_id = int(results_path.split("_")[-1].strip(".pkl"))
        
        # Load corresponding small stimulus image file
        image_path = results_path.replace("small_stimulus", "big_stimulus").replace(".pkl", ".npy")
        with open(image_path, "rb") as f:
            image = np.load(f)
        
        # Load corresponding big stimulus results
        big_results_path = f"results/coen_cagli_big_stimulus_results_{stimulus_id}.pkl"
        with open(big_results_path, "rb") as f:
            big_results = pickle.load(f)
            big_X = big_results['posterior']['X'].data
            big_x_mean = big_X.mean((0, 1))
            big_G = big_results['posterior']['G'].data
            big_g_mean = big_G.mean((0, 1))
        
        # Append to results dictionary
        results_dict['stimulus_id'].append(stimulus_id)
        results_dict['big_stimulus'].append(image)
        results_dict['small_x_mean'].append(X_mean)
        results_dict['small_g_mean'].append(G_mean)
        results_dict['big_x_mean'].append(big_x_mean)
        results_dict['big_g_mean'].append(big_g_mean)
        
        print(f"Processed {results_path} with stimulus_id {stimulus_id}")

# Convert to pandas DataFrame
df = pd.DataFrame(results_dict)

# Add center mean columns
G_dim = 10
center_ids = list(range(G_dim*4, G_dim*4 + G_dim))  # [40, 41, 42, ..., 49]

print(f"Center IDs: {center_ids}")
print(f"This corresponds to indices {G_dim*4} to {G_dim*4 + G_dim - 1}")

# Extract the center region for each small_x_mean and compute its mean
df['small_x_center_mean'] = df['small_x_mean'].apply(lambda x: x[center_ids].mean())

# Extract the center region for each big_x_mean and compute its mean
df['big_x_center_mean'] = df['big_x_mean'].apply(lambda x: x[center_ids].mean())

# %%
df

# %%
# Create a dictionary to store the top 200 stimulus IDs for each center neuron
top_activating_stimuli = {}

# For each center neuron
for i, center_neuron_idx in enumerate(center_ids):
    # Extract the activation values for this specific center neuron across all stimuli
    neuron_activations = df['small_x_mean'].apply(lambda x: x[center_neuron_idx])
    
    # Get the indices of the top 200 most activating stimuli
    top_200_indices = neuron_activations.nlargest(200).index
    
    # Get the corresponding stimulus IDs
    top_200_stimulus_ids = df.loc[top_200_indices, 'stimulus_id'].tolist()
    
    # Store in dictionary
    neuron_name = f'center_neuron_{i}'  # neuron 0-9
    top_activating_stimuli[neuron_name] = top_200_stimulus_ids
    
    print(f"Center neuron {i} (index {center_neuron_idx}): Top 200 stimulus IDs")
    print(f"  Max activation: {neuron_activations.max():.4f}")
    print(f"  Min activation in top 200: {neuron_activations.iloc[top_200_indices].min():.4f}")
    print(f"  Sample stimulus IDs: {top_200_stimulus_ids[:10]}")

print(f"Created dictionary with {len(top_activating_stimuli)} center neurons")
print(f"Each neuron has {len(top_activating_stimuli['center_neuron_0'])} top activating stimulus IDs")

# %%
top_activating_stimuli

# %%
# Create a more organized DataFrame for analysis
top_stimuli_data = []

for neuron_name, stimulus_ids in top_activating_stimuli.items():
    neuron_idx = int(neuron_name.split('_')[-1])  # Extract neuron index (0-9)
    center_neuron_pos = center_ids[neuron_idx]  # Get the actual position in x_mean (40-49)
    
    for rank, stimulus_id in enumerate(stimulus_ids):
        # Get the activation value for this stimulus and neuron
        row_idx = df[df['stimulus_id'] == stimulus_id].index[0]
        
        # Current center neuron activation
        small_center_x_mean = df.loc[row_idx, 'small_x_mean'][center_neuron_pos]
        big_center_x_mean = df.loc[row_idx, 'big_x_mean'][center_neuron_pos]
        
        # Get full arrays for small and big stimulus
        small_x_full = df.loc[row_idx, 'small_x_mean']
        big_x_full = df.loc[row_idx, 'big_x_mean']
        small_g_full = df.loc[row_idx, 'small_g_mean']
        big_g_full = df.loc[row_idx, 'big_g_mean']
        
        # All center neurons (include all 10 center neurons)
        small_all_center_array = small_x_full[center_ids]
        big_all_center_array = big_x_full[center_ids]
        
        # All neurons (include all neurons, preserving order)
        small_all_other_array = small_x_full
        big_all_other_array = big_x_full
        
        # All G neurons
        small_g_array = small_g_full
        big_g_array = big_g_full
        
        top_stimuli_data.append({
            'neuron_idx': neuron_idx,
            'center_neuron_pos': center_neuron_pos,
            'stimulus_id': stimulus_id,
            'rank': rank + 1,  # 1-indexed rank
            'small_x_center': small_center_x_mean,
            'big_x_center': big_center_x_mean,
            'small_x_all_center_array': small_all_center_array,
            'big_x_all_center_array': big_all_center_array,
            'small_x_all_other_array': small_all_other_array,
            'big_x_all_other_array': big_all_other_array,
            'small_g_array': small_g_array,
            'big_g_array': big_g_array
        })

# Create DataFrame
top_stimuli_df = pd.DataFrame(top_stimuli_data)

# %%
top_stimuli_df

# %%
# Analysis: Compare small vs big stimulus center activations
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the difference between big and small center activations
top_stimuli_df['center_activation_ratio'] = top_stimuli_df['big_x_center'] / top_stimuli_df['small_x_center']

# Summary statistics
print("=== Small vs Big Stimulus Center Activation Analysis ===")
print(f"Total number of stimulus-neuron pairs: {len(top_stimuli_df)}")
print()

print("Summary Statistics:")
print(f"Small stimulus center activation - Mean: {top_stimuli_df['small_x_center'].mean():.4f}, Std: {top_stimuli_df['small_x_center'].std():.4f}")
print(f"Big stimulus center activation - Mean: {top_stimuli_df['big_x_center'].mean():.4f}, Std: {top_stimuli_df['big_x_center'].std():.4f}")
print(f"Activation ratio (big / small) - Mean: {top_stimuli_df['center_activation_ratio'].mean():.4f}, Std: {top_stimuli_df['center_activation_ratio'].std():.4f}")
print()

# Check how many cases have higher/lower activation with big stimulus
higher_with_big = (top_stimuli_df['big_x_center'] > top_stimuli_df['small_x_center']).sum()
lower_with_big = (top_stimuli_df['big_x_center'] < top_stimuli_df['small_x_center']).sum()
equal = (top_stimuli_df['big_x_center'] == top_stimuli_df['small_x_center']).sum()

print("Comparison Results:")
print(f"Cases where big stimulus has HIGHER activation: {higher_with_big} ({100*higher_with_big/len(top_stimuli_df):.1f}%)")
print(f"Cases where big stimulus has LOWER activation: {lower_with_big} ({100*lower_with_big/len(top_stimuli_df):.1f}%)")
print(f"Cases where activations are EQUAL: {equal} ({100*equal/len(top_stimuli_df):.1f}%)")
print()

# Analysis by neuron
print("Analysis by Center Neuron:")
neuron_analysis = top_stimuli_df.groupby('neuron_idx').agg({
    'small_x_center': ['mean', 'std'],
    'big_x_center': ['mean', 'std'],
    'center_activation_ratio': ['mean', 'std']
}).round(4)

print(neuron_analysis)

# %%
# Visualizations: Small vs Big Stimulus Center Activations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Scatter plot: Small vs Big activation
axes[0, 0].scatter(top_stimuli_df['small_x_center'], top_stimuli_df['big_x_center'], 
                   alpha=0.6, s=20)
axes[0, 0].plot([0, top_stimuli_df['small_x_center'].max()], 
                [0, top_stimuli_df['small_x_center'].max()], 'r--', alpha=0.8, label='y=x')
axes[0, 0].set_xlabel('Small Stimulus Center Activation')
axes[0, 0].set_ylabel('Big Stimulus Center Activation')
axes[0, 0].set_title('Small vs Big Stimulus Center Activation')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution of activation differences
axes[0, 1].hist(top_stimuli_df['center_activation_diff'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8, label='No difference')
axes[0, 1].axvline(top_stimuli_df['center_activation_diff'].mean(), color='orange', 
                   linestyle='-', alpha=0.8, label=f'Mean = {top_stimuli_df["center_activation_diff"].mean():.4f}')
axes[0, 1].set_xlabel('Activation Difference (Big - Small)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Activation Differences')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot by neuron
neuron_data = []
neuron_labels = []
for neuron_idx in sorted(top_stimuli_df['neuron_idx'].unique()):
    neuron_subset = top_stimuli_df[top_stimuli_df['neuron_idx'] == neuron_idx]
    neuron_data.append(neuron_subset['center_activation_diff'])
    neuron_labels.append(f'Neuron {neuron_idx}')

axes[1, 0].boxplot(neuron_data, labels=neuron_labels)
axes[1, 0].set_xlabel('Center Neuron')
axes[1, 0].set_ylabel('Activation Difference (Big - Small)')
axes[1, 0].set_title('Activation Differences by Center Neuron')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.8)

# 4. Activation ratio distribution
axes[1, 1].hist(top_stimuli_df['center_activation_ratio'], bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(1, color='red', linestyle='--', alpha=0.8, label='Ratio = 1 (equal)')
axes[1, 1].axvline(top_stimuli_df['center_activation_ratio'].mean(), color='orange', 
                   linestyle='-', alpha=0.8, label=f'Mean = {top_stimuli_df["center_activation_ratio"].mean():.4f}')
axes[1, 1].set_xlabel('Activation Ratio (Big / Small)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Activation Ratios')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%


# %%
# Calculate Modulation Ratio (MR) and Geometric Mean
# MR = big_response / small_response (same as center_activation_ratio)
# MR < 1: surround suppression
# MR > 1: facilitation

# Filter for images that evoked a measurable response (non-zero small stimulus activation)
measurable_responses = top_stimuli_df[top_stimuli_df['small_x_center'] > 0].copy()

print("=== Modulation Ratio (MR) Analysis ===")
print(f"Total stimulus-neuron pairs: {len(top_stimuli_df)}")
print(f"Pairs with measurable response (small_x_center > 0): {len(measurable_responses)}")
print(f"Percentage with measurable response: {100 * len(measurable_responses) / len(top_stimuli_df):.1f}%")
print()

if len(measurable_responses) > 0:
    # Calculate geometric mean of MR
    from scipy import stats
    
    # Geometric mean using scipy
    geometric_mean_mr = stats.gmean(measurable_responses['center_activation_ratio'])
    
    # Alternative calculation: exp(mean(log(MR)))
    geometric_mean_mr_alt = np.exp(np.log(measurable_responses['center_activation_ratio']).mean())
    
    print("Modulation Ratio (MR) Statistics:")
    print(f"Arithmetic mean MR: {measurable_responses['center_activation_ratio'].mean():.4f}")
    print(f"Geometric mean MR: {geometric_mean_mr:.4f}")
    print(f"Geometric mean MR (alternative calc): {geometric_mean_mr_alt:.4f}")
    print(f"Median MR: {measurable_responses['center_activation_ratio'].median():.4f}")
    print()
    
    # Classification of responses
    suppression = (measurable_responses['center_activation_ratio'] < 1).sum()
    facilitation = (measurable_responses['center_activation_ratio'] > 1).sum()
    no_change = (measurable_responses['center_activation_ratio'] == 1).sum()
    
    print("Response Classification:")
    print(f"Surround suppression (MR < 1): {suppression} ({100*suppression/len(measurable_responses):.1f}%)")
    print(f"Facilitation (MR > 1): {facilitation} ({100*facilitation/len(measurable_responses):.1f}%)")
    print(f"No change (MR = 1): {no_change} ({100*no_change/len(measurable_responses):.1f}%)")
    print()
    
    print(f"*** The geometric mean of the MR for natural images that evoked a measurable response was {geometric_mean_mr:.4f} ***")
    
    if geometric_mean_mr < 1:
        print("This indicates overall SURROUND SUPPRESSION when enlarging stimuli.")
    elif geometric_mean_mr > 1:
        print("This indicates overall FACILITATION when enlarging stimuli.")
    else:
        print("This indicates NO NET EFFECT when enlarging stimuli.")
        
else:
    print("No measurable responses found!")

# %%
# Verify that big_stimulus files with and without "presented" are identical
import numpy as np
import glob
import os

# Find all big_stimulus files (both regular and presented versions)
regular_files = glob.glob("results/coen_cagli_big_stimulus_*.npy")
presented_files = glob.glob("results/coen_cagli_presented_big_stimulus_*.npy")

print(f"Found {len(regular_files)} regular big_stimulus files")
print(f"Found {len(presented_files)} presented big_stimulus files")

# Extract stimulus IDs from both sets
def extract_stimulus_id(filename):
    # Extract number from filename like "coen_cagli_big_stimulus_123.npy" or "coen_cagli_presented_big_stimulus_123.npy"
    basename = os.path.basename(filename)
    return int(basename.split('_')[-1].replace('.npy', ''))

regular_ids = set([extract_stimulus_id(f) for f in regular_files])
presented_ids = set([extract_stimulus_id(f) for f in presented_files])

# Find common stimulus IDs (those that have both versions)
common_ids = regular_ids.intersection(presented_ids)
print(f"Found {len(common_ids)} stimulus IDs that have both regular and presented versions")

if len(common_ids) > 0:
    print(f"Sample common IDs: {sorted(list(common_ids))[:10]}")
    
    # Compare a few files to verify they are identical
    comparison_results = []
    test_ids = sorted(list(common_ids))
    
    for stimulus_id in test_ids:
        regular_file = f"results/coen_cagli_big_stimulus_{stimulus_id}.npy"
        presented_file = f"results/coen_cagli_presented_big_stimulus_{stimulus_id}.npy"
        
        # Load both files
        regular_data = np.load(regular_file)
        presented_data = np.load(presented_file)
        
        # Check if they are identical
        are_identical = np.array_equal(regular_data, presented_data)
        
        # Check shapes and dtypes
        same_shape = regular_data.shape == presented_data.shape
        same_dtype = regular_data.dtype == presented_data.dtype

        print(f"Stimulus ID {stimulus_id}: Identical = {are_identical}, ")

# %%
# Analysis: G neuron responses to small vs big stimuli
# Understanding if inhibitory G neurons drive the center X neuron effects

print("=== G Neuron (Inhibitory) Response Analysis ===")
print(f"Total stimulus-neuron pairs: {len(top_stimuli_df)}")
print()

# Add G neuron mean activations to our analysis DataFrame
top_stimuli_df['small_g_mean'] = top_stimuli_df['small_g_array'].apply(lambda x: x.mean())
top_stimuli_df['big_g_mean'] = top_stimuli_df['big_g_array'].apply(lambda x: x.mean())
top_stimuli_df['g_activation_diff'] = top_stimuli_df['big_g_mean'] - top_stimuli_df['small_g_mean']
top_stimuli_df['g_activation_ratio'] = top_stimuli_df['big_g_mean'] / top_stimuli_df['small_g_mean']

# Summary statistics for G neurons
print("G Neuron Summary Statistics:")
print(f"Small stimulus G activation - Mean: {top_stimuli_df['small_g_mean'].mean():.4f}, Std: {top_stimuli_df['small_g_mean'].std():.4f}")
print(f"Big stimulus G activation - Mean: {top_stimuli_df['big_g_mean'].mean():.4f}, Std: {top_stimuli_df['big_g_mean'].std():.4f}")
print(f"G activation difference (big - small) - Mean: {top_stimuli_df['g_activation_diff'].mean():.4f}, Std: {top_stimuli_df['g_activation_diff'].std():.4f}")
print(f"G activation ratio (big / small) - Mean: {top_stimuli_df['g_activation_ratio'].mean():.4f}, Std: {top_stimuli_df['g_activation_ratio'].std():.4f}")
print()

# Compare G neuron changes vs X neuron changes
higher_g_with_big = (top_stimuli_df['big_g_mean'] > top_stimuli_df['small_g_mean']).sum()
lower_g_with_big = (top_stimuli_df['big_g_mean'] < top_stimuli_df['small_g_mean']).sum()
equal_g = (top_stimuli_df['big_g_mean'] == top_stimuli_df['small_g_mean']).sum()

print("G Neuron Response Direction:")
print(f"Cases where big stimulus has HIGHER G activation: {higher_g_with_big} ({100*higher_g_with_big/len(top_stimuli_df):.1f}%)")
print(f"Cases where big stimulus has LOWER G activation: {lower_g_with_big} ({100*lower_g_with_big/len(top_stimuli_df):.1f}%)")
print(f"Cases where G activations are EQUAL: {equal_g} ({100*equal_g/len(top_stimuli_df):.1f}%)")
print()

# Correlation analysis between G and X changes
from scipy.stats import pearsonr

# Correlation between G activation changes and X activation changes
g_x_corr, g_x_p = pearsonr(top_stimuli_df['g_activation_diff'], top_stimuli_df['center_activation_diff'])
print(f"Correlation between G activation change and X center activation change: r = {g_x_corr:.4f}, p = {g_x_p:.6f}")

# Correlation between G ratio and X ratio
g_x_ratio_corr, g_x_ratio_p = pearsonr(top_stimuli_df['g_activation_ratio'], top_stimuli_df['center_activation_ratio'])
print(f"Correlation between G activation ratio and X center activation ratio: r = {g_x_ratio_corr:.4f}, p = {g_x_ratio_p:.6f}")
print()



# Geometric mean analysis for G neurons (like we did for X neurons)
measurable_g_responses = top_stimuli_df[top_stimuli_df['small_g_mean'] > 0].copy()
print(f"G neuron pairs with measurable response: {len(measurable_g_responses)} ({100 * len(measurable_g_responses) / len(top_stimuli_df):.1f}%)")

if len(measurable_g_responses) > 0:
    from scipy import stats
    geometric_mean_g_mr = stats.gmean(measurable_g_responses['g_activation_ratio'])
    
    print(f"G neuron geometric mean MR: {geometric_mean_g_mr:.4f}")
    
    g_suppression = (measurable_g_responses['g_activation_ratio'] < 1).sum()
    g_facilitation = (measurable_g_responses['g_activation_ratio'] > 1).sum()
    
    print(f"G neuron suppression (MR < 1): {g_suppression} ({100*g_suppression/len(measurable_g_responses):.1f}%)")
    print(f"G neuron facilitation (MR > 1): {g_facilitation} ({100*g_facilitation/len(measurable_g_responses):.1f}%)")
    
    if geometric_mean_g_mr > 1:
        print("✅ G neurons show FACILITATION with larger stimuli - increased inhibition")
        print("   This could explain X neuron suppression via increased inhibitory drive")
    elif geometric_mean_g_mr < 1:
        print("✅ G neurons show SUPPRESSION with larger stimuli - decreased inhibition") 
        print("   This would lead to X neuron facilitation via disinhibition")
    
    print()
    print("=== MECHANISTIC INTERPRETATION ===")
    x_geom_mean = stats.gmean(top_stimuli_df[top_stimuli_df['small_x_center'] > 0]['center_activation_ratio'])
    
    if geometric_mean_g_mr > 1 and x_geom_mean < 1:
        print("🎯 SURROUND SUPPRESSION MECHANISM:")
        print("   Larger stimuli → Increased G neuron activity → Stronger inhibition → Suppressed X neurons")
    elif geometric_mean_g_mr < 1 and x_geom_mean > 1:
        print("🎯 SURROUND FACILITATION MECHANISM:")
        print("   Larger stimuli → Decreased G neuron activity → Weaker inhibition → Facilitated X neurons")
    else:
        print("🤔 Mixed or complex relationship between G and X neuron responses")
        
    print(f"   X center geometric mean MR: {x_geom_mean:.4f}")
    print(f"   G neuron geometric mean MR: {geometric_mean_g_mr:.4f}")

# %%
top_stimuli_df

# %%
# Spatial Analysis: G neuron responses at center positions vs other positions
# Compare G neuron at same position as center X neuron vs other G neurons

print("=== Spatial G Neuron Analysis ===")
print("Comparing G neurons at center positions vs other positions")
print()

# Add columns for spatially-specific G neuron analysis
spatial_g_data = []

for idx, row in top_stimuli_df.iterrows():
    neuron_idx = row['neuron_idx']  # This is 0-9 (center neuron index)
    
    # Center X neurons are at positions 40-49, corresponding to row 4, columns 0-9
    # The corresponding G neuron would be at the same spatial position
    corresponding_g_idx = neuron_idx  # G neuron at same column as center X neuron
    
    # Extract G neuron activations
    small_g_array = row['small_g_array']  # All 10 G neurons
    big_g_array = row['big_g_array']      # All 10 G neurons
    
    # Same-position G neuron (at same column as center X neuron)
    small_g_same_pos = small_g_array[corresponding_g_idx]
    big_g_same_pos = big_g_array[corresponding_g_idx]
    
    # Other G neurons (all except the same-position one)
    other_g_indices = [i for i in range(len(small_g_array)) if i != corresponding_g_idx]
    small_g_others_mean = small_g_array[other_g_indices].mean()
    big_g_others_mean = big_g_array[other_g_indices].mean()
    
    # Calculate ratios and differences
    g_same_pos_ratio = big_g_same_pos / small_g_same_pos if small_g_same_pos > 0 else np.nan
    g_others_ratio = big_g_others_mean / small_g_others_mean if small_g_others_mean > 0 else np.nan
    
    g_same_pos_diff = big_g_same_pos - small_g_same_pos
    g_others_diff = big_g_others_mean - small_g_others_mean
    
    spatial_g_data.append({
        'neuron_idx': neuron_idx,
        'stimulus_id': row['stimulus_id'],
        'rank': row['rank'],
        'corresponding_g_idx': corresponding_g_idx,
        'small_g_same_pos': small_g_same_pos,
        'big_g_same_pos': big_g_same_pos,
        'small_g_others_mean': small_g_others_mean,
        'big_g_others_mean': big_g_others_mean,
        'g_same_pos_ratio': g_same_pos_ratio,
        'g_others_ratio': g_others_ratio,
        'g_same_pos_diff': g_same_pos_diff,
        'g_others_diff': g_others_diff,
        'center_x_ratio': row['center_activation_ratio']
    })

# Add spatial data to main DataFrame
for i, data in enumerate(spatial_g_data):
    for key, value in data.items():
        if key not in ['neuron_idx', 'stimulus_id', 'rank']:  # Don't overwrite existing columns
            top_stimuli_df.loc[i, key] = value

print("=== Same-Position G Neuron Analysis ===")
print("(G neuron at same spatial position as center X neuron)")

# Filter for measurable responses
measurable_same_pos = top_stimuli_df[top_stimuli_df['small_g_same_pos'] > 0].copy()
print(f"Measurable same-position G responses: {len(measurable_same_pos)} ({100*len(measurable_same_pos)/len(top_stimuli_df):.1f}%)")

if len(measurable_same_pos) > 0:
    print(f"Small stimulus same-pos G - Mean: {measurable_same_pos['small_g_same_pos'].mean():.4f}")
    print(f"Big stimulus same-pos G - Mean: {measurable_same_pos['big_g_same_pos'].mean():.4f}")
    print(f"Same-pos G ratio (big/small) - Mean: {measurable_same_pos['g_same_pos_ratio'].mean():.4f}")
    
    # Geometric mean for same-position G neurons
    geom_mean_same_pos = stats.gmean(measurable_same_pos['g_same_pos_ratio'])
    print(f"Same-pos G geometric mean ratio: {geom_mean_same_pos:.4f}")

print()
print("=== Other G Neurons Analysis ===")
print("(Average of all other G neurons, excluding same-position)")

measurable_others = top_stimuli_df[top_stimuli_df['small_g_others_mean'] > 0].copy()
print(f"Measurable other G responses: {len(measurable_others)} ({100*len(measurable_others)/len(top_stimuli_df):.1f}%)")

if len(measurable_others) > 0:
    print(f"Small stimulus other G - Mean: {measurable_others['small_g_others_mean'].mean():.4f}")
    print(f"Big stimulus other G - Mean: {measurable_others['big_g_others_mean'].mean():.4f}")
    print(f"Other G ratio (big/small) - Mean: {measurable_others['g_others_ratio'].mean():.4f}")
    
    # Geometric mean for other G neurons
    geom_mean_others = stats.gmean(measurable_others['g_others_ratio'])
    print(f"Other G geometric mean ratio: {geom_mean_others:.4f}")

print()
print("=== Spatial Comparison ===")

if len(measurable_same_pos) > 0 and len(measurable_others) > 0:
    # Compare same-position vs others
    print(f"Same-position G ratio: {geom_mean_same_pos:.4f}")
    print(f"Other G neurons ratio: {geom_mean_others:.4f}")
    
    if geom_mean_same_pos > geom_mean_others:
        print("Same-position G neurons show STRONGER facilitation than other G neurons")
        print("   → More localized inhibitory response at center")
    elif geom_mean_same_pos < geom_mean_others:
        print("Same-position G neurons show WEAKER facilitation than other G neurons")
        print("   → Less localized, more diffuse inhibitory response")
    else:
        print("Same-position and other G neurons respond similarly")
        print("   → Uniform inhibitory response across G neurons")


# %%
top_stimuli_df

# %%
# Diagnostic: Why is the geometric mean 0.0000?
print("=== Diagnostic Analysis for G Neuron Geometric Mean ===")

# Check the g_same_pos_ratio values
print("G same-position ratio statistics:")
print(f"Total values: {len(top_stimuli_df)}")
print(f"Non-null values: {top_stimuli_df['g_same_pos_ratio'].notna().sum()}")
print(f"Values == 0: {(top_stimuli_df['g_same_pos_ratio'] == 0).sum()}")
print(f"Values < 0.001: {(top_stimuli_df['g_same_pos_ratio'] < 0.001).sum()}")
print(f"Values == inf: {np.isinf(top_stimuli_df['g_same_pos_ratio']).sum()}")
print()

# Check small and big G same position values
print("Small G same-position values:")
print(f"Mean: {top_stimuli_df['small_g_same_pos'].mean():.6f}")
print(f"Min: {top_stimuli_df['small_g_same_pos'].min():.6f}")
print(f"Max: {top_stimuli_df['small_g_same_pos'].max():.6f}")
print(f"Values == 0: {(top_stimuli_df['small_g_same_pos'] == 0).sum()}")
print(f"Values < 0.001: {(top_stimuli_df['small_g_same_pos'] < 0.001).sum()}")
print()

print("Big G same-position values:")
print(f"Mean: {top_stimuli_df['big_g_same_pos'].mean():.6f}")
print(f"Min: {top_stimuli_df['big_g_same_pos'].min():.6f}")
print(f"Max: {top_stimuli_df['big_g_same_pos'].max():.6f}")
print(f"Values == 0: {(top_stimuli_df['big_g_same_pos'] == 0).sum()}")
print(f"Values < 0.001: {(top_stimuli_df['big_g_same_pos'] < 0.001).sum()}")
print()

# Look at some sample ratios
print("Sample g_same_pos_ratio values:")
sample_ratios = top_stimuli_df['g_same_pos_ratio'].dropna().head(20)
for i, ratio in enumerate(sample_ratios):
    print(f"  {i}: {ratio:.6f}")
print()

# Check if the issue is with very small numbers
non_zero_ratios = top_stimuli_df[top_stimuli_df['g_same_pos_ratio'] > 0]['g_same_pos_ratio']
print(f"Non-zero ratios count: {len(non_zero_ratios)}")
if len(non_zero_ratios) > 0:
    print(f"Min non-zero ratio: {non_zero_ratios.min():.10f}")
    print(f"Max non-zero ratio: {non_zero_ratios.max():.10f}")
    
    # Try geometric mean on non-zero values only
    if len(non_zero_ratios) > 0:
        geom_mean_nonzero = stats.gmean(non_zero_ratios)
        print(f"Geometric mean of non-zero ratios: {geom_mean_nonzero:.6f}")

# Check the measurable_same_pos filtering
print()
print("Measurable same-position filtering:")
print(f"Original length: {len(top_stimuli_df)}")
measurable_test = top_stimuli_df[top_stimuli_df['small_g_same_pos'] > 0]
print(f"After filtering small_g_same_pos > 0: {len(measurable_test)}")

if len(measurable_test) > 0:
    print("Ratios in measurable subset:")
    print(f"Min ratio: {measurable_test['g_same_pos_ratio'].min():.10f}")
    print(f"Max ratio: {measurable_test['g_same_pos_ratio'].max():.10f}")
    print(f"Zeros in measurable subset: {(measurable_test['g_same_pos_ratio'] == 0).sum()}")
    
    # Try different threshold for "measurable"
    measurable_test_strict = top_stimuli_df[top_stimuli_df['small_g_same_pos'] > 0.001]
    print(f"After filtering small_g_same_pos > 0.001: {len(measurable_test_strict)}")
    
    if len(measurable_test_strict) > 0:
        geom_mean_strict = stats.gmean(measurable_test_strict['g_same_pos_ratio'])
        print(f"Geometric mean with stricter filter: {geom_mean_strict:.6f}")

# %%
# Fix the spatial G neuron analysis
print("=== Fixed Spatial G Neuron Analysis ===")
print("The issue was with the array indexing assumption")
print()

# Check the actual structure of G neuron arrays
sample_g_array = top_stimuli_df.iloc[0]['small_g_array']
print(f"G neuron array shape: {sample_g_array.shape}")
print(f"G neuron array length: {len(sample_g_array)}")
print(f"G neuron array min: {sample_g_array.min():.6f}")
print(f"G neuron array max: {sample_g_array.max():.6f}")
print()

# The G neurons are arranged in a 10x1 array (10 G neurons total)
# Center X neurons are at positions 40-49 in the X array (90 total X neurons)
# But G neurons are only 10 total, so we need a different mapping

print("=== Corrected Spatial Analysis ===")

# Clear the problematic columns and recalculate
top_stimuli_df = top_stimuli_df.drop(columns=[
    'corresponding_g_idx', 'small_g_same_pos', 'big_g_same_pos', 
    'small_g_others_mean', 'big_g_others_mean', 'g_same_pos_ratio', 
    'g_others_ratio', 'g_same_pos_diff', 'g_others_diff', 'center_x_ratio'
], errors='ignore')

# Recalculate spatial G analysis with correct indexing
spatial_g_data_fixed = []

for idx, row in top_stimuli_df.iterrows():
    neuron_idx = row['neuron_idx']  # This is 0-9 (center neuron index)
    
    # G neurons: there are 10 G neurons total (indices 0-9)
    # We can use the neuron_idx directly since both are 0-9
    corresponding_g_idx = neuron_idx  # G neuron at same index as center X neuron
    
    # Extract G neuron activations
    small_g_array = row['small_g_array']  # All 10 G neurons
    big_g_array = row['big_g_array']      # All 10 G neurons
    
    # Same-index G neuron (at same index as center X neuron)
    small_g_same_idx = small_g_array[corresponding_g_idx]
    big_g_same_idx = big_g_array[corresponding_g_idx]
    
    # Other G neurons (all except the same-index one)
    other_g_indices = [i for i in range(len(small_g_array)) if i != corresponding_g_idx]
    small_g_others_mean = small_g_array[other_g_indices].mean()
    big_g_others_mean = big_g_array[other_g_indices].mean()
    
    # Calculate ratios and differences
    g_same_idx_ratio = big_g_same_idx / small_g_same_idx if small_g_same_idx > 0 else np.nan
    g_others_ratio = big_g_others_mean / small_g_others_mean if small_g_others_mean > 0 else np.nan
    
    g_same_idx_diff = big_g_same_idx - small_g_same_idx
    g_others_diff = big_g_others_mean - small_g_others_mean
    
    # Add to DataFrame
    top_stimuli_df.loc[idx, 'corresponding_g_idx'] = corresponding_g_idx
    top_stimuli_df.loc[idx, 'small_g_same_idx'] = small_g_same_idx
    top_stimuli_df.loc[idx, 'big_g_same_idx'] = big_g_same_idx
    top_stimuli_df.loc[idx, 'small_g_others_mean'] = small_g_others_mean
    top_stimuli_df.loc[idx, 'big_g_others_mean'] = big_g_others_mean
    top_stimuli_df.loc[idx, 'g_same_idx_ratio'] = g_same_idx_ratio
    top_stimuli_df.loc[idx, 'g_others_ratio'] = g_others_ratio
    top_stimuli_df.loc[idx, 'g_same_idx_diff'] = g_same_idx_diff
    top_stimuli_df.loc[idx, 'g_others_diff'] = g_others_diff

print("=== Same-Index G Neuron Analysis (Fixed) ===")

# Filter for measurable responses
measurable_same_idx = top_stimuli_df[top_stimuli_df['small_g_same_idx'] > 0].copy()
print(f"Measurable same-index G responses: {len(measurable_same_idx)} ({100*len(measurable_same_idx)/len(top_stimuli_df):.1f}%)")

if len(measurable_same_idx) > 0:
    print(f"Small stimulus same-idx G - Mean: {measurable_same_idx['small_g_same_idx'].mean():.4f}")
    print(f"Big stimulus same-idx G - Mean: {measurable_same_idx['big_g_same_idx'].mean():.4f}")
    print(f"Same-idx G ratio (big/small) - Mean: {measurable_same_idx['g_same_idx_ratio'].mean():.4f}")
    
    # Check for zeros in the ratios
    zero_ratios = (measurable_same_idx['g_same_idx_ratio'] == 0).sum()
    print(f"Zero ratios in measurable subset: {zero_ratios}")
    
    # Geometric mean for same-index G neurons (excluding zeros)
    non_zero_same_idx = measurable_same_idx[measurable_same_idx['g_same_idx_ratio'] > 0]
    if len(non_zero_same_idx) > 0:
        geom_mean_same_idx = stats.gmean(non_zero_same_idx['g_same_idx_ratio'])
        print(f"Same-idx G geometric mean ratio (excluding zeros): {geom_mean_same_idx:.4f}")
        print(f"Based on {len(non_zero_same_idx)} non-zero ratios out of {len(measurable_same_idx)} total")

print()
print("=== Other G Neurons Analysis (Fixed) ===")

measurable_others = top_stimuli_df[top_stimuli_df['small_g_others_mean'] > 0].copy()
print(f"Measurable other G responses: {len(measurable_others)} ({100*len(measurable_others)/len(top_stimuli_df):.1f}%)")

if len(measurable_others) > 0:
    print(f"Small stimulus other G - Mean: {measurable_others['small_g_others_mean'].mean():.4f}")
    print(f"Big stimulus other G - Mean: {measurable_others['big_g_others_mean'].mean():.4f}")
    print(f"Other G ratio (big/small) - Mean: {measurable_others['g_others_ratio'].mean():.4f}")
    
    # Geometric mean for other G neurons
    non_zero_others = measurable_others[measurable_others['g_others_ratio'] > 0]
    if len(non_zero_others) > 0:
        geom_mean_others = stats.gmean(non_zero_others['g_others_ratio'])
        print(f"Other G geometric mean ratio (excluding zeros): {geom_mean_others:.4f}")
        print(f"Based on {len(non_zero_others)} non-zero ratios out of {len(measurable_others)} total")

# %%



