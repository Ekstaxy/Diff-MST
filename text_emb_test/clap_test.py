import numpy as np
import torch
import laion_clap
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def int16_to_float32(x):
    """Convert int16 audio to float32."""
    return (x / 32768.0).astype('float32')

def float32_to_int16(x):
    """Convert float32 audio to int16."""
    x = np.clip(x, a_min=-1.0, a_max=1.)
    return (x * 32768.).astype('int16')

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()

with open("text_bright.txt", "r") as f:
    text = f.readlines()
    text = [t.strip() for t in text]
text1_batch = [text[i:i+50] for i in range(0, len(text), 50)]

with open("text_dark.txt", "r") as f:
    text2 = f.readlines()
    text2 = [t.strip() for t in text2]
text2_batch = [text2[i:i+50] for i in range(0, len(text2), 50)]
    
with open("text_bass_enhance.txt", "r") as f:
    text3 = f.readlines()
    text3 = [t.strip() for t in text3]
text3_batch = [text3[i:i+50] for i in range(0, len(text3), 50)]

    
with open("text_vocal_enhance.txt", "r") as f:
    text4 = f.readlines()
    text4 = [t.strip() for t in text4]
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
print(f"X3 shape: {X3.shape}, X4 shape: {X4.shape}")
X = np.vstack((X1, X2, X3, X4))
# X = np.vstack((X3, X4))
y = np.array([0]*len(X1) + [1]*len(X2) + [2]*len(X3) + [3]*len(X4))
# y = np.array([2]*len(X3) + [3]*len(X4))

tsne = manifold.TSNE(n_components=2, init='pca', random_state=42, early_exaggeration=14.0, max_iter=5000, perplexity=45)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title('t-SNE Visualization of Text Embeddings')
plt.savefig('bass_vocal_enhance.png')