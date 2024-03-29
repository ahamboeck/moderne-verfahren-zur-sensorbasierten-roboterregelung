import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the eigenvalues
data = pd.read_csv('eigenvalues.csv', header=None)
components = data[0].to_numpy()  # Convert to NumPy array for compatibility
eigenvalues = data[1].to_numpy()  # Convert to NumPy array for compatibility

# Calculate the percentage of variance explained by each component
total_variance = sum(eigenvalues)
variance_explained = [(i / total_variance) * 100 for i in eigenvalues]
cumulative_variance_explained = np.cumsum(variance_explained)

# Find the number of components where cumulative variance explained exceeds 95%
num_components_95_variance = np.argmax(cumulative_variance_explained >= 95) + 1

# Create the scree plot
plt.figure(figsize=(10, 6))

# Plot the individual explained variances
plt.bar(components, variance_explained, alpha=0.5, align='center', label='Individual explained variance')

# Plot the cumulative explained variance
plt.step(components, cumulative_variance_explained, where='mid', label='Cumulative explained variance')

plt.ylabel('Explained variance ratio (%)')
plt.xlabel('Principal Component')
plt.legend(loc='best')
plt.title('Scree Plot')
plt.axhline(y=95, color='r', linestyle='--', label='95% Explained Variance')
plt.axvline(x=num_components_95_variance, color='g', linestyle='--', label='Optimal Components')

# Print the number of components on the plot
plt.text(num_components_95_variance + 0.5, 90, f'95% variance\nat {num_components_95_variance} components', color='green', verticalalignment='top')

plt.legend()
plt.tight_layout()

plt.savefig('scree_plot.png')  # Save the plot to a file