import numpy as np
import torch
import laion_clap
import matplotlib.pyplot as plt
from sklearn import manifold

def int16_to_float32(x):
    return (x / 32768.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.)
    return (x * 32768.).astype('int16')

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()

# ==========================================
# 1. Spatial Analysis (Wide, Narrow, Left, Right)
# ==========================================

with open("text_wide.txt", "r") as f:
    text1 = [t.strip() for t in f.readlines() if t.strip()]
text1_batch = [text1[i:i+50] for i in range(0, len(text1), 50)]

with open("text_narrow.txt", "r") as f:
    text2 = [t.strip() for t in f.readlines() if t.strip()]
text2_batch = [text2[i:i+50] for i in range(0, len(text2), 50)]

with open("text_left.txt", "r") as f:
    text3 = [t.strip() for t in f.readlines() if t.strip()]
text3_batch = [text3[i:i+50] for i in range(0, len(text3), 50)]

with open("text_right.txt", "r") as f:
    text4 = [t.strip() for t in f.readlines() if t.strip()]
text4_batch = [text4[i:i+50] for i in range(0, len(text4), 50)]

X1 = np.zeros((0, 512))
for t in text1_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X1 = np.concatenate((X1, emb), axis=0)
X1 = X1[1:, :]  # Remove the first zero row added during initialization

X2 = np.zeros((0, 512))
for t in text2_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X2 = np.concatenate((X2, emb), axis=0)
X2 = X2[1:, :]  # Remove the first zero row added during initialization

X3 = np.zeros((0, 512))
for t in text3_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X3 = np.concatenate((X3, emb), axis=0)
X3 = X3[1:, :]  # Remove the first zero row added during initialization

X4 = np.zeros((0, 512))
for t in text4_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X4 = np.concatenate((X4, emb), axis=0)
X4 = X4[1:, :]  # Remove the first zero row added during initialization

X = np.vstack((X1, X2, X3, X4))
y = np.array([0]*len(X1) + [1]*len(X2) + [2]*len(X3) + [3]*len(X4))

tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, early_exaggeration=14.0, max_iter=5000, perplexity=30)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

plt.figure(figsize=(8, 8))
labels = ['Wide', 'Narrow', 'Left', 'Right']
colors = ['blue', 'cyan', 'green', 'lime']

for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=colors[y[i]],
             fontdict={'weight': 'bold', 'size': 9})

# Add legend manually to match style
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=colors[i], label=f"{i}: {labels[i]}") for i in range(4)]
plt.legend(handles=patches)

plt.xticks([])
plt.yticks([])
plt.title('t-SNE: Spatial (Wide/Narrow/Left/Right)')
plt.savefig('spatial_tsne.png')
plt.close()

# ==========================================
# 2. Dynamics & Gain Analysis (Punchy, Compressed, High Gain, Low Gain)
# ==========================================

with open("text_punchy.txt", "r") as f:
    text1 = [t.strip() for t in f.readlines() if t.strip()]
text1_batch = [text1[i:i+50] for i in range(0, len(text1), 50)]

with open("text_compressed.txt", "r") as f:
    text2 = [t.strip() for t in f.readlines() if t.strip()]
text2_batch = [text2[i:i+50] for i in range(0, len(text2), 50)]

with open("text_high_gain.txt", "r") as f:
    text3 = [t.strip() for t in f.readlines() if t.strip()]
text3_batch = [text3[i:i+50] for i in range(0, len(text3), 50)]

with open("text_low_gain.txt", "r") as f:
    text4 = [t.strip() for t in f.readlines() if t.strip()]
text4_batch = [text4[i:i+50] for i in range(0, len(text4), 50)]

X1 = np.zeros((0, 512))
for t in text1_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X1 = np.concatenate((X1, emb), axis=0)
X1 = X1[1:, :]  # Remove the first zero row added during initialization

X2 = np.zeros((0, 512))
for t in text2_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X2 = np.concatenate((X2, emb), axis=0)
X2 = X2[1:, :]  # Remove the first zero row added during initialization

X3 = np.zeros((0, 512))
for t in text3_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X3 = np.concatenate((X3, emb), axis=0)
X3 = X3[1:, :]  # Remove the first zero row added during initialization

X4 = np.zeros((0, 512))
for t in text4_batch:
    emb = clap.get_text_embedding(t, use_tensor=True).cpu().detach().numpy()
    X4 = np.concatenate((X4, emb), axis=0)
X4 = X4[1:, :]  # Remove the first zero row added during initialization

X = np.vstack((X1, X2, X3, X4))
y = np.array([0]*len(X1) + [1]*len(X2) + [2]*len(X3) + [3]*len(X4))

tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, early_exaggeration=14.0, max_iter=5000, perplexity=30)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

plt.figure(figsize=(8, 8))
labels = ['Punchy', 'Compressed', 'High Gain', 'Low Gain']
colors = ['orange', 'red', 'purple', 'pink']

for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=colors[y[i]],
             fontdict={'weight': 'bold', 'size': 9})

patches = [mpatches.Patch(color=colors[i], label=f"{i}: {labels[i]}") for i in range(4)]
plt.legend(handles=patches)

plt.xticks([])
plt.yticks([])
plt.title('t-SNE: Dynamics & Gain')
plt.savefig('dynamics_tsne.png')