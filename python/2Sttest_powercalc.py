"""
Audiology clinical trial sample size calculation

Script computes the required sample size for a clinical trial testing the effectiveness 
of a new hearing aid, with a two-sample t-test.
Primary outcome measure = improvement in speech recognition scores (%). 
Expected improvement over conventional hearing aids = effect_size%
Known standard deviation = std_dev% based on prior studies.
"""

import math

def compute_sample_size(effect_size, std_dev, alpha=0.05, power=0.80):
    # Z values for given alpha and power
    Z_alpha_2 = 1.96  # for alpha = 0.05 (two-tailed)
    Z_beta = 0.84  # for power = 0.80
    
    # Formula for sample size for a two-sample t-test
    n = ((Z_alpha_2 + Z_beta) * std_dev / effect_size) ** 2
    return math.ceil(n)  # round up to ensure enough participants

# Parameters
effect_size = 0.10  # Expected difference in speech recognition scores
std_dev = 0.15  # Standard deviation based on prior studies

# Compute sample size
sample_size = compute_sample_size(effect_size, std_dev)

print(f"Required sample size per group: {sample_size}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(effect_size, std_dev):
    # Generate data for the two groups
    conventional_scores = np.random.normal(0, std_dev, 1000)
    new_hearing_aid_scores = np.random.normal(effect_size, std_dev, 1000)

    # Plot distributions
    sns.kdeplot(conventional_scores, label="Conventional Hearing Aid", shade=True)
    sns.kdeplot(new_hearing_aid_scores, label="New Hearing Aid", shade=True)
    sns.set_style("whitegrid")
    plt.title("Distribution of Speech Recognition Scores", weight='bold', fontsize=14)
    plt.xlabel("Speech Recognition Score Improvement (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

plot_distributions(effect_size, std_dev)