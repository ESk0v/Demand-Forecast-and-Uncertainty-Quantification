import torch
import sys
import os


def main(local=False, line_idx=0):
    """
    Inspect the dataset by printing encoder, decoder, and target for a given sample index.

    Args:
        local: If True, use relative paths. If False, use server paths.
        line_idx: The sample index to inspect.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    if local:
        _dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(_dir, "..", "Files", "dataset.pt")
    else:
        dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"

    # -----------------------------
    # Load dataset
    # -----------------------------
    data = torch.load(dataset_path, weights_only=False)

    encoder = data['encoder']
    decoder = data['decoder']
    target = data['target']

    def print_full_tensor_line(tensor, name, idx):
        print(f"\n{name} - line {idx}:")
        arr = tensor[idx].numpy() if isinstance(tensor[idx], torch.Tensor) else tensor[idx]
        if arr.ndim == 2:
            for i, row in enumerate(arr):
                print(f"Row {i}: {row}")
        else:
            print(arr)

    print_full_tensor_line(encoder, "Encoder", line_idx)
    print_full_tensor_line(decoder, "Decoder", line_idx)
    print_full_tensor_line(target, "Target", line_idx)

    print(f"\nEncoder shape: {encoder.shape}")
    print(f"Decoder shape: {decoder.shape}")
    print(f"Target shape:  {target.shape}")


# Allow standalone execution: python3 DatasetLookup.py [line_index]
if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # Default to local when running standalone
    main(local=True, line_idx=idx)
