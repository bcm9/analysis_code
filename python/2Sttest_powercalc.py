# Sample size calculation for two sample t-test
# see https://www.youtube.com/watch?v=WzRW-MTx0Ps
import statsmodels.stats.api as sms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set parameters
effect_size = 0.2 # d
alpha = 0.05
power = 0.95 # 1-beta
std_dev = 1  # Standard deviation
ratio = 1 # Sample ratio allocation ratio

# Calculate sample size for independent samples t-test
sample_size = sms.tt_ind_solve_power(effect_size, alpha=alpha, power=power, ratio=ratio, alternative='two-sided')

if ratio > 1:
    print("Required sample size group 1:", round(sample_size))
    print("Required sample size group 2:", round(sample_size)*ratio)
else:
    print("Required sample size in each group:", round(sample_size))


# Generate random data for demonstration
# Assuming Group 1 has mean 0 and Group 2 has mean 0+effect_size*std_dev
group1 = np.random.normal(0, std_dev, round(sample_size))
group2 = np.random.normal(effect_size*std_dev, std_dev, round(sample_size))

# Plot KDE distribution
sns.kdeplot(group1, label="Group 1", shade=True)
sns.kdeplot(group2, label="Group 2", shade=True)
plt.title('KDE plot for two groups')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()