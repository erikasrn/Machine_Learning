import tensorflow as tf

@tf.function
def compute_new_centroids(data, cluster_assignments, centroids, penalty_threshold, penalty_factor):
    num_clusters = centroids.shape[0]
    new_centroids = []

    for c in range(num_clusters):
        # Get points in the current cluster
        cluster_points = tf.gather(data, tf.where(cluster_assignments == c)[:, 0])
        
        if tf.shape(cluster_points)[0] == 0:  # Handle empty clusters
            new_centroids.append(centroids[c])
            continue

        # Calculate distances of points to the current centroid
        cluster_distances = tf.norm(cluster_points - centroids[c], axis=1)

        # Apply penalty: Weight decreases for distant points
        weights = tf.maximum(1.0 - penalty_factor * tf.maximum(cluster_distances - penalty_threshold, 0), 0.01)
        weighted_sum = tf.reduce_sum(cluster_points * weights[:, None], axis=0)
        weight_sum = tf.reduce_sum(weights)

        # Calculate new centroid
        new_centroid = weighted_sum / weight_sum
        new_centroids.append(new_centroid)

    return tf.stack(new_centroids)

def tensorflow_kmeans(data, num_clusters, num_iterations=100, penalty_threshold=0.3, penalty_factor=0.5):
    """
    Custom K-means clustering implementation using TensorFlow with soft penalties for distant points.
    """
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    initial_centroids = tf.random.shuffle(data)[:num_clusters]
    centroids = tf.Variable(initial_centroids)

    for _ in range(num_iterations):
        # Compute distances from points to centroids (Euclidean distance)
        distances = tf.norm(data[:, None] - centroids, axis=2)

        # Assign each data point to the cluster of the nearest centroid
        cluster_assignments = tf.argmin(distances, axis=1)

        # Update centroids with soft penalties
        centroids.assign(compute_new_centroids(data, cluster_assignments, centroids, penalty_threshold, penalty_factor))

    return centroids.numpy(), cluster_assignments.numpy()
