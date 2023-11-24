# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from numpy.random import default_rng
from sklearn.metrics.pairwise import cosine_similarity

rng = default_rng(seed=192553)


# %%
def custom_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# %%
def validate_cosine_similarity(a, b):
    x = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
    y = custom_cosine_similarity(a, b)
    print("custom", x)
    print("sklearn", y)
    return np.isclose(x, y)


# %%
a = rng.random(100000)
b = rng.random(100000)
print(
    np.isclose(
        cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)),
        custom_cosine_similarity(a, b),
    )
)

# %%
raw_exc_stimuli = np.load(
    "/src/project/data/experiment/exc_images.npy", allow_pickle=True
)
raw_exc_stimuli = np.array([np.array(x) for x in raw_exc_stimuli])
print(raw_exc_stimuli.shape)
raw_exc_stimuli_flat = raw_exc_stimuli.reshape(raw_exc_stimuli.shape[0], -1)
print(raw_exc_stimuli_flat.shape)
a = raw_exc_stimuli_flat[0]
b = raw_exc_stimuli_flat[1]
print(a.shape)
print(b.shape)

# %%
print(validate_cosine_similarity(a, b))

# %%
normed_a = a / np.linalg.norm(a)
normed_b = b / np.linalg.norm(b)
print(validate_cosine_similarity(normed_a, normed_b))

# %%
raw_exc_stimuli = np.array([np.array(x) for x in raw_exc_stimuli])

# %%
raw_exc_stimuli.shape

# %%
raw_exc_stimuli_flat = raw_exc_stimuli.reshape(raw_exc_stimuli.shape[0], -1)

# %%
raw_exc_stimuli_flat.shape

# %%
first_sample = raw_exc_stimuli_flat[0]
second_sample = raw_exc_stimuli_flat[1]
print(first_sample.shape)
print(second_sample.shape)

# %%
print(
    np.isclose(
        cosine_similarity(first_sample.reshape(1, -1), second_sample.reshape(1, -1)),
        custom_cosine_similarity(first_sample, second_sample),
    )
)

# %%
cosine_similarity(first_sample.reshape(1, -1), second_sample.reshape(1, -1))

# %%
custom_cosine_similarity(first_sample, second_sample)

# %%
fig, ax = plt.subplots()
sns.histplot(first_sample, ax=ax, bins=100, color="blue")
sns.histplot(second_sample, ax=ax, bins=100, color="red")

# %%
second_sample

# %%
# construct two orthogonal vectors
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
print(
    np.isclose(
        cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)),
        custom_cosine_similarity(a, b),
    )
)

# %%
scipy.sparse.issparse(raw_exc_stimuli_flat)

# %%
normalized_first_sample = first_sample / np.linalg.norm(first_sample)
normalized_second_sample = second_sample / np.linalg.norm(second_sample)
sklearn_cosine_similarity = cosine_similarity(
    normalized_first_sample.reshape(1, -1), normalized_second_sample.reshape(1, -1)
)[0, 0]
custom_cosine_similarity = custom_cosine_similarity(
    normalized_first_sample, normalized_second_sample
)
print(np.isclose(sklearn_cosine_similarity, custom_cosine_similarity))

# %%
