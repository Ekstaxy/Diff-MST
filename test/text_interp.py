import torch
import argparse
import laion_clap
import numpy as np

from utils import linear_interpolation, spherical_linear_interpolation

def parse_args():
    parser = argparse.ArgumentParser(description="Test text interpolation between prompts.")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the model checkpoint."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(args.ckpt_path)
    model.eval()

    text_prompts = [
        "mute", "quiet", "loud", "blaring" 
    ]
    edge_prompts = ["mute", "blaring"]
    alphas = [1.0, 1.0, 1.0, 1.0]

    text_embeddings = model.get_text_embedding(text_prompts)
    edge_embeddings = model.get_text_embedding(edge_prompts)
    left_emb = text_embeddings[0]
    right_emb = text_embeddings[-1]

    left_similarity = np.dot(left_emb, edge_embeddings[0]) / (np.linalg.norm(left_emb) * np.linalg.norm(edge_embeddings[0]))
    right_similarity = np.dot(right_emb, edge_embeddings[1]) / (np.linalg.norm(right_emb) * np.linalg.norm(edge_embeddings[1]))
    edge_similarity = np.dot(edge_embeddings[0], edge_embeddings[1]) / (np.linalg.norm(edge_embeddings[0]) * np.linalg.norm(edge_embeddings[1]))
    print(f"\nLeft edge similarity: {left_similarity:.4f}")
    print(f"Right edge similarity: {right_similarity:.4f}")
    print(f"Edge similarity: {edge_similarity:.4f}\n")

    lin_embeddings = [
        linear_interpolation(left_emb, right_emb, alpha) for alpha in alphas
    ]
    slerp_embeddings = [
        spherical_linear_interpolation(left_emb, right_emb, alpha) for alpha in alphas
    ]

    # Calculate cosine similarities to verify interpolation correctness
    lin_similarities = []
    slerp_similarities = []

    for i in range(len(text_embeddings)):
        lin_sim = np.dot(
            lin_embeddings[i], text_embeddings[i]
        ) / (np.linalg.norm(lin_embeddings[i]) * np.linalg.norm(text_embeddings[i]))
        slerp_sim = np.dot(
            slerp_embeddings[i], text_embeddings[i]
        ) / (np.linalg.norm(slerp_embeddings[i]) * np.linalg.norm(text_embeddings[i]))
        lin_similarities.append(lin_sim)
        slerp_similarities.append(slerp_sim)

    print("Linear Interpolation Cosine Similarities:")
    for alpha, sim in zip(alphas, lin_similarities):
        print(f"Alpha: {alpha:.2f}, Similarity: {sim:.4f}")
    
    print("\nSpherical Linear Interpolation Cosine Similarities:")
    for alpha, sim in zip(alphas, slerp_similarities):
        print(f"Alpha: {alpha:.2f}, Similarity: {sim:.4f}")

if __name__ == "__main__":
    main()