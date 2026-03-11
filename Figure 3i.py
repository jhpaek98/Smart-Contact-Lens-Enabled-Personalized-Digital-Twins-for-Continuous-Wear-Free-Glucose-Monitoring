# PCA Explained variance ratio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_path = "input.xlsx"
folder_path = os.path.dirname(file_path)

df = pd.read_excel(file_path)

sample_names = df.iloc[:, 0]
X = df.drop(columns=df.columns[0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

plot_path = os.path.join(folder_path, "pca_variance_plot.png")

plt.figure(figsize=(8,5))
plt.bar(range(1,4), explained, alpha=0.7)
plt.plot(range(1,4), cum_explained, marker='o')
plt.axhline(0.8, linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Top 3 Principal Components')
plt.xticks(range(1,4))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

pca_scores_df = pd.DataFrame(X_pca, columns=['PC1','PC2','PC3'])
pca_scores_df.insert(0, 'Sample', sample_names)

variance_df = pd.DataFrame({
    'PC':['PC1','PC2','PC3'],
    'Explained Variance Ratio': explained,
    'Cumulative Variance': cum_explained
})

excel_path = os.path.join(folder_path, "PCA_results.xlsx")

with pd.ExcelWriter(excel_path) as writer:
    pca_scores_df.to_excel(writer, sheet_name="PCA_scores", index=False)
    variance_df.to_excel(writer, sheet_name="PCA_variance", index=False)

print(excel_path)