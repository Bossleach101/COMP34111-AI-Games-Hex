import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys

# Add current directory to path so we can import BoardEvaluator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BoardEvaluator import HexNet, HexDataset

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = HexNet().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "hex_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found at {model_path}. Please train the model first.")
        return

    # Load Validation Data
    val_data_path = os.path.join(os.path.dirname(__file__), "data/validation_11x11_games.json")
    if not os.path.exists(val_data_path):
        print(f"Validation data not found at {val_data_path}")
        return

    try:
        val_dataset = HexDataset(val_data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Evaluating on {len(val_dataset)} samples...")

    model.eval()
    total_loss = 0
    correct_sign = 0
    total_samples = 0

    with torch.no_grad():
        for data, target_policy, target_value in val_loader:
            data, target_value = data.to(device), target_value.to(device)
            
            output_policy, output_value = model(data)
            
            # Value Loss (MSE)
            loss = F.mse_loss(output_value.view(-1), target_value, reduction='sum')
            total_loss += loss.item()
            
            # Accuracy (Sign of value)
            # Target is 1 or -1 (or close to it). Output is tanh (-1 to 1).
            # We check if signs match.
            preds = output_value.view(-1)
            # Handle cases where target might be 0 (draw) if any, though Hex usually has no draws.
            # Assuming targets are non-zero.
            correct_sign += ((preds > 0) == (target_value > 0)).sum().item()
            
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_sign / total_samples

    print(f"Validation Results:")
    print(f"Average MSE Loss: {avg_loss:.6f}")
    print(f"Accuracy (Sign): {accuracy:.4f} ({correct_sign}/{total_samples})")

if __name__ == "__main__":
    evaluate()
