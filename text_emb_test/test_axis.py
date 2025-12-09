import numpy as np
import torch
import laion_clap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

def int16_to_float32(x):
    return (x / 32768.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.)
    return (x * 32768.).astype('int16')

# 載入模型
clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()

def run_axis_analysis(file_a, file_b, label_a, label_b, color_a, color_b, title, filename):
    print(f"Running Analysis: {title}...")
    
    # Load Data A
    try:
        with open(file_a, "r") as f:
            text1 = [t.strip() for t in f.readlines() if t.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_a}, skipping.")
        return

    text1_batch = [text1[i:i+50] for i in range(0, len(text1), 50)]

    # Load Data B
    try:
        with open(file_b, "r") as f:
            text2 = [t.strip() for t in f.readlines() if t.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_b}, skipping.")
        return

    text2_batch = [text2[i:i+50] for i in range(0, len(text2), 50)]

    # Compute Embeddings A
    X1 = np.zeros((0, 512))
    for t in text1_batch:
        emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
        X1 = np.concatenate((X1, emb), axis=0)
    X1 = X1[1:, :]

    # Compute Embeddings B
    X2 = np.zeros((0, 512))
    for t in text2_batch:
        emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
        X2 = np.concatenate((X2, emb), axis=0)
    X2 = X2[1:, :]

    # Combine
    X = np.vstack((X1, X2))
    y = np.array([0]*len(X1) + [1]*len(X2))

    # PCA Projection (Find the main axis of difference)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Calculate Cosine Similarity between Centroids
    centroid_a = np.mean(X1, axis=0)
    centroid_b = np.mean(X2, axis=0)
    
    # Cosine Similarity Formula: (A . B) / (||A|| * ||B||)
    cos_sim = np.dot(centroid_a, centroid_b) / (np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b))
    print(f"  > Cosine Similarity between {label_a} and {label_b}: {cos_sim:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    
    # Plot Scatter
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c=color_a, label=label_a, alpha=0.7)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c=color_b, label=label_b, alpha=0.7)

    # Draw arrow between centroids in PCA space
    cent_pca_a = np.mean(X_pca[y==0], axis=0)
    cent_pca_b = np.mean(X_pca[y==1], axis=0)
    plt.arrow(cent_pca_a[0], cent_pca_a[1], 
              cent_pca_b[0]-cent_pca_a[0], cent_pca_b[1]-cent_pca_a[1], 
              color='black', width=0.002, head_width=0.02, length_includes_head=True, alpha=0.5)
    
    plt.title(f'{title}\nCentroid Similarity: {cos_sim:.3f}')
    plt.xlabel('PC1 (Main Axis of Difference)')
    plt.ylabel('PC2 (Secondary Variation)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(filename)
    plt.close()
    print(f"  > Saved to {filename}\n")

# ==========================================
# Execute 4 Separate "Battles"
# ==========================================

# 1. Position Axis (Left vs Right)
run_axis_analysis(
    "text_left.txt", "text_right.txt", 
    "Left", "Right", 
    "green", "lime", 
    "Spatial Position Axis", "axis_pos.png"
)

# 2. Width Axis (Wide vs Narrow)
run_axis_analysis(
    "text_wide.txt", "text_narrow.txt", 
    "Wide", "Narrow", 
    "blue", "cyan", 
    "Spatial Width Axis", "axis_width.png"
)

# 3. Dynamics Axis (Punchy vs Compressed)
run_axis_analysis(
    "text_punchy.txt", "text_compressed.txt", 
    "Punchy", "Compressed", 
    "orange", "red", 
    "Dynamics Axis", "axis_dyn.png"
)

# 4. Gain Axis (High vs Low)
# Note: Using filenames generated in previous turn. 
# If you used 'text_loud.txt', change 'text_high_gain.txt' below.
run_axis_analysis(
    "text_high_gain.txt", "text_low_gain.txt", 
    "High Gain", "Low Gain", 
    "purple", "pink", 
    "Gain/Loudness Axis", "axis_gain.png"
)