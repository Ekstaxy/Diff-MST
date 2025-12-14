import numpy as np
import torch
import laion_clap
import matplotlib.pyplot as plt
from sklearn import decomposition

def int16_to_float32(x):
    """Convert int16 audio to float32."""
    return (x / 32768.0).astype('float32')

def float32_to_int16(x):
    """Convert float32 audio to int16."""
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32768.0).astype('int16')

def slerp(p0, p1, t):
    """Spherical linear interpolation between two vectors."""
    p0_norm = p0 / np.linalg.norm(p0)
    p1_norm = p1 / np.linalg.norm(p1)
    dot = np.clip(np.dot(p0_norm, p1_norm), -1.0, 1.0)
    theta = np.arccos(dot) * t
    relative_vec = p1_norm - dot * p0_norm
    relative_vec /= np.linalg.norm(relative_vec)
    return p0_norm * np.cos(theta) + relative_vec * np.sin(theta)

def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()

# text = ["Strongly make it darker", "Make it a bit darker", "Slightly make it darker", "Don't make it darker",
#         "Don't make it brighter", "Slightly make it brighter", "Make it a bit brighter", "Strongly make it brighter"]
text = ["Strongly enhance the bass", "Enhance the bass a bit", "Slightly enhance the bass", "Don't enhance the bass",
        "Don't enhance the vocals", "Slightly enhance the vocals", "Enhance the vocals a bit", "Strongly enhance the vocals"]
X = clap.get_text_embedding(list(text), use_tensor=True).cpu().detach().numpy()

t_values = np.linspace(0, 1, num=8)
Y = [slerp(X[0], X[-1], t) for t in t_values]

# Calculate cosine similarity between X and Y
cosine_similarities = [cosine_similarity(X[i], Y[i]) for i in range(min(len(X), len(Y)))]

# Print the cosine similarities
for i, cos_sim in enumerate(cosine_similarities):
    print(f"Cosine similarity between X[{i}] and Y[{i}]: {cos_sim}")
    
    # Plot cosine similarities vs interpolation t
    plt.figure(figsize=(8, 4))
    x = t_values[:len(cosine_similarities)]
    plt.plot(x, cosine_similarities, marker='o', linestyle='-', color='C0')
    for xi, yi in zip(x, cosine_similarities):
        plt.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    plt.xlabel('Interpolation t')
    plt.ylabel('Cosine similarity')
    plt.title('Cosine similarity between original and slerp-interpolated embeddings')
    plt.grid(True)
    plt.ylim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig('slerp_cosine_similarity_bass&vocal.png')