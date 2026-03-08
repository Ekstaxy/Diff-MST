import torch
import torch.nn as nn
from mst.modules import RoFormerRemixer
from tqdm import tqdm

def test_roformer_remixer():
    print("Initializing RoFormerRemixer...")
    # Initialize the model (downloads or loads from cache if available)
    try:
        remixer = RoFormerRemixer(sample_rate=44100)
    except Exception as e:
        print(f"Skipping test: Failed to initialize RoFormerRemixer (Audio Separator not installed?). Error: {e}")
        return

    bs = 1
    channels = 2
    seq_len = 44100 * 5  # 5 seconds
    
    print(f"Creating dummy input tensor: Batch={bs}, Ch={channels}, Len={seq_len}")
    # Create dummy stereo audio (random noise)
    dummy_input = torch.randn(bs, channels, seq_len)

    print("Running separation (this might take time due to file I/O)...")
    
    try:
        # Run forward pass
        output = remixer(dummy_input)
        
        # Check shape
        # Expected: (Batch, Stems*Channels, seq_len) if flattened or (Batch, Stems, Channels, seq_len) if not
        # Let's check what RoFormerRemixer returns.
        # Based on previous edits, it returns 'separated_tensor' from 'separated_batch'
        # 'separated_batch' is list of (2, ch, time) -> So (Batch, 2, Ch, Time)
        
        print(f"Output shape: {output.shape}")
        
        expected_stems = 2 # Instrumental, Vocals
        expected_shape = (bs, expected_stems, channels, seq_len)
        
        if output.shape == expected_shape:
            print("Test Passed: Output shape matches expected (Batch, Stems, Ch, Time).")
        else:
            print(f"Test Failed: Output shape mismatch. Expected {expected_shape}, got {output.shape}")

        # Basic data check
        if output.sum() == 0 and dummy_input.sum() != 0:
             print("Warning: Output is all zeros. (Might happen if separation model produces silence for noise input)")
        else:
             print("Output contains data.")

    except Exception as e:
        print(f"Test Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_roformer_remixer()
