import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Utility function to evaluate clustering
def evaluate_clustering(model_name, labels, y_true):
    print(f"\nModel: {model_name}")
    print("Adjusted Rand Index (how well clustering matches labels):", adjusted_rand_score(y_true, labels))
    if len(set(labels)) > 1:
        print("Silhouette Score:", silhouette_score(X_scaled, labels))
    else:
        print("Silhouette Score: Not available (only one cluster)")

# --- K-Means Clustering ---
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
evaluate_clustering("K-Means", kmeans_labels, y)

# --- Agglomerative Clustering ---
agglo = AgglomerativeClustering(n_clusters=2)
agglo_labels = agglo.fit_predict(X_scaled)
evaluate_clustering("Agglomerative Clustering", agglo_labels, y)

# --- DBSCAN ---
dbscan = DBSCAN(eps=1.8, min_samples=5)  # eps might need tuning
dbscan_labels = dbscan.fit_predict(X_scaled)
evaluate_clustering("DBSCAN", dbscan_labels, y)

# --- Optional: Plot clusters in 2D PCA space ---
def plot_clusters(X_pca, labels, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

plot_clusters(X_pca, kmeans_labels, "K-Means Clusters (PCA)")
plot_clusters(X_pca, agglo_labels, "Agglomerative Clusters (PCA)")
plot_clusters(X_pca, dbscan_labels, "DBSCAN Clusters (PCA)")
