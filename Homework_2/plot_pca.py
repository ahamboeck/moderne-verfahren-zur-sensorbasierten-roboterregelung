import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the eigenvalues
data = pd.read_csv('eigenvalues.csv', header=None)
components = data[0]
eigenvalues = data[1]

# Calculate the percentage of variance explained by each component
total_variance = sum(eigenvalues)
variance_explained = [(i / total_variance) * 100 for i in eigenvalues]
cumulative_variance_explained = np.cumsum(variance_explained)

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
plt.axvline(x=np.argmax(cumulative_variance_explained > 95) + 1, color='g', linestyle='--', label='Optimal Components')

plt.legend()
plt.tight_layout()
plt.savefig('scree_plot.png')  # Save the plot to a file
