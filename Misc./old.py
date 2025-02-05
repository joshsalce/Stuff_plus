# Cluster sizes and ground truth sizes
cluster_labels = list(sorted_values)
cluster_sizes = list(sorted_counts)
ground_truth_sizes = list(df_hue.value_counts().values)
ground_truth_labels = list(df_hue.value_counts().index)

print({cluster_labels[i]: cluster_sizes[i] for i in range(len(cluster_labels))})
print({ground_truth_labels[i]: ground_truth_sizes[i] for i in range(len(ground_truth_labels))})

# Create a dictionary to store the mapping between cluster labels and ground truth labels
cluster_to_ground_truth_mapping = {}

# Initialize an empty matrix to store absolute differences
differences_matrix = np.zeros((len(ground_truth_sizes),len(cluster_sizes)))

# Calculate absolute differences and fill the matrix
for i in range(len(ground_truth_sizes)):
    for j in range(len(cluster_sizes)):
        differences_matrix[i][j] = abs(ground_truth_sizes[i] - cluster_sizes[j])

#print(differences_matrix)
        
while len(cluster_sizes) > 0:
    # Initialize variables to track the best match
    best_cluster_index = None
    best_ground_truth_index = None
    
    # Find the row index with the smallest value (minimum across all rows)
    min_row_index = np.argmin(differences_matrix)
    
    # Calculate the row and column index based on the flattened index
    min_row, min_col = np.unravel_index(min_row_index, differences_matrix.shape)
    
    #print(min_row, min_col)
    cluster_to_ground_truth_mapping[cluster_labels[min_col]] = ground_truth_labels[min_row]
    del cluster_labels[min_col]
    del cluster_sizes[min_col]
    del ground_truth_sizes[min_row]
    del ground_truth_labels[min_row]
    
    #print(cluster_to_ground_truth_mapping)
    # Delete the row with the smallest value
    differences_matrix = np.delete(differences_matrix, min_row, axis=0)
    # Delete column
    differences_matrix = np.delete(differences_matrix, min_col, axis=1)
    #print(differences_matrix)


print("Cluster to Ground Truth Mapping:", cluster_to_ground_truth_mapping)



#-----------------------------------------------------------------------------

# Visualize the results (for 2D PCA, assuming n_components=2)
plt.figure(figsize=(4, 4))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k','brown','lightgreen']

#for i in range(num_clusters):
for i in range(num_clusters):
    sns.scatterplot(data=df_pca[df['ClusterLabel'] == i], x='PC1', y='PC2', 
                    label=f'Cluster {i}', color=colors[i], alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA and GMM Clustering', fontsize=16)
plt.legend(title='Cluster Label')
plt.show()