import numpy as np
from scipy.special import i0


def vonmises_logpdf(x, mu, kappa):
    return kappa * np.cos(x - mu) / (2 * np.pi * i0(kappa))


def rate_code(
    baseline_firing_rate=0,
    global_orientation_variable=None,
    orientation_preferences=None,
    vonmises_loc=0,
    vonmises_kappa=1,
    rate_code_signature="base",
):
    if rate_code_signature == "base":
        lam = baseline_firing_rate
    elif rate_code_signature == "base+vonmises":
        lam = baseline_firing_rate + vonmises_logpdf(
            global_orientation_variable - orientation_preferences,
            vonmises_loc,
            vonmises_kappa,
        )
    elif rate_code_signature == "base+inv_vonmises":
        lam = baseline_firing_rate + 1 / vonmises_logpdf(
            global_orientation_variable - orientation_preferences,
            vonmises_loc,
            vonmises_kappa,
        )
    elif rate_code_signature == "vonmises":
        lam = vonmises_logpdf(
            global_orientation_variable - orientation_preferences,
            vonmises_loc,
            vonmises_kappa,
        )
    elif rate_code_signature == "inv_vonmises":
        lam = 1 / vonmises_logpdf(
            global_orientation_variable - orientation_preferences,
            vonmises_loc,
            vonmises_kappa,
        )
    else:
        raise ValueError("Unknown rate code signature")
    return lam
