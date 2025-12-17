import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from agents.Group11.utils import HexPlanes

class HexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super(HexConv2d, self).__init__()

        # Standard 2D Convolution
        # We assume kernel_size=3 for a standard hex neighborhood
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

        # Create the binary mask
        # Shape: (out_channels, in_channels, 3, 3)
        self.register_buffer('mask', torch.ones_like(self.conv.weight))

        # Zero out the "non-hex" corners.
        # Note: Which corners you zero depends on how you skew your board.
        # Common Axial skew ignores Top-Left (0,0) and Bottom-Right (2,2)
        self.mask[:, :, 0, 0] = 0
        self.mask[:, :, 2, 2] = 0

    def forward(self, x):
        # Apply the mask to the weights
        # We multiply the weights by the mask to ensure the "dead" corners
        # always stay zero and have zero gradient.
        masked_weight = self.conv.weight * self.mask

        return F.conv2d(x, masked_weight, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)

class HexNet(nn.Module):
    def __init__(self):
        super(HexNet, self).__init__()
        self.conv1 = HexConv2d(4, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = HexConv2d(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = HexConv2d(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = HexConv2d(64, 64)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Policy Head
        self.p_conv = HexConv2d(64, 2)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 11 * 11, 121)
        
        # Value Head
        self.v_conv = HexConv2d(64, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * 11 * 11, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Policy
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(-1, 2 * 11 * 11)
        p = self.p_fc(p)
        # p = F.log_softmax(p, dim=1) # Return logits
        
        # Value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(-1, 1 * 11 * 11)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return p, v

class HexDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Pre-calculate valid next moves
        self.next_move_indices = np.full(len(self.data), -1, dtype=np.int32)
        
        # We can iterate and check stone counts to find boundaries
        # This assumes data is sequential
        print("Processing dataset for move targets...")
        for i in range(len(self.data) - 1):
            # Quick check: if next state has exactly 1 more stone, it's a valid next move
            # We can optimize by just checking stone counts without full array creation if possible,
            # but here we load the full json so we have the lists.
            # Accessing list of lists is fast.
            
            # Heuristic: Check if next board has +1 stone count
            # Flattening to count non-zeros
            curr_board = self.data[i]['position']
            next_board = self.data[i+1]['position']
            
            # Count stones (sum of absolute values or just non-zeros)
            # Since values are 0, 1, -1, we can just count non-zeros
            curr_stones = sum(1 for row in curr_board for x in row if x != 0)
            next_stones = sum(1 for row in next_board for x in row if x != 0)
            
            if next_stones == curr_stones + 1:
                self.next_move_indices[i] = i + 1
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        board = np.array(item['position'], dtype=np.int8)
        outcome = item['outcome']
        
        # Infer current player
        count_1 = np.sum(board == 1)
        count_neg_1 = np.sum(board == -1)
        current_player = 1 if count_1 == count_neg_1 else -1
        
        features = HexPlanes.get_all_feature_planes(board, current_player)
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        outcome = torch.tensor(outcome, dtype=torch.float32)
        
        # Policy Target
        next_idx = self.next_move_indices[idx]
        if next_idx != -1:
            next_board = np.array(self.data[next_idx]['position'], dtype=np.int8)
            diff = next_board - board
            # Find the index of the move
            # diff will have exactly one non-zero element (1 or -1)
            move_indices = np.argwhere(diff != 0)
            if len(move_indices) == 1:
                r, c = move_indices[0]
                target_move = r * 11 + c
            else:
                # Should not happen if stone count check passed, but safety fallback
                target_move = -100
        else:
            target_move = -100 # Ignore index for CrossEntropyLoss
            
        policy_target = torch.tensor(target_move, dtype=torch.long)
        
        return features, policy_target, outcome

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target_policy, target_value) in enumerate(train_loader):
        data, target_policy, target_value = data.to(device), target_policy.to(device), target_value.to(device)
        optimizer.zero_grad()
        output_policy, output_value = model(data)
        
        # Loss
        # Value loss: MSE
        loss_v = F.mse_loss(output_value.view(-1), target_value)
        
        # Policy loss: CrossEntropy
        # target_policy contains indices of the move (0-120) or -100 to ignore
        loss_p = F.cross_entropy(output_policy, target_policy, ignore_index=-100)
        
        loss = loss_v + loss_p
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tLoss_P: {loss_p.item():.6f}\tLoss_V: {loss_v.item():.6f}')

class HexModelInference:
    """
    Class to handle model inference for Hex game states.
    """
    def __init__(self, modelpath):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.model = HexNet()
        
        # Use weights_only=True for security and to silence warnings (PyTorch 2.0+)
        # This is supported in torch 2.5.1
        self.model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
        
        self.model.to(device)
        self.model.eval()
        
        # Removed JIT compilation as it was causing segmentation faults in the Docker environment
        self.device = device

    def predict(self, board_state, current_player):
        """
        Predicts the policy and value for a given board state.
        
        Args:
            board_state: 11x11 numpy array (0=Empty, 1=Black, -1=White)
            current_player: The value representing the current player (e.g., 1 or -1)
        
        Returns:
            policy: numpy array of shape (121,) representing move probabilities
            value: float representing the expected outcome for the current player
        """
        feature_planes = HexPlanes.get_all_feature_planes(board_state, current_player)
        input_tensor = torch.from_numpy(feature_planes).unsqueeze(0).float().to(self.device)  # Shape: (1, 4, 11, 11)

        with torch.no_grad():
            output_policy, output_value = self.model(input_tensor)
            
        # Apply softmax to policy logits to get probabilities
        policy_probs = F.softmax(output_policy, dim=1)
        
        policy = policy_probs.cpu().numpy().reshape(11, 11)  # Shape: (11, 11)
        value = output_value.cpu().item()  # Scalar
        
        return policy, value


def train2():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Path to data
    data_path = os.path.join(os.path.dirname(__file__), 'data/train_11x11_games.json')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    dataset = HexDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = HexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        
    # Save model
    save_path = os.path.join(os.path.dirname(__file__), "hex_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    board = np.zeros((11, 11), dtype=int)
    board[4,5] = 1
    board[4,7] = 1
    board[4,8] = -1
    board[4,9] = 1

    board[5,2] = -1
    board[5,5] = -1
    board[5,7] = -1
    board[5,8] = -1
    board[5,9] = 1

    board[6,3] = -1
    board[6,6] = -1
    board[6,7] = 1
    board[6,8] = -1
    board[6,9] = 1

    board[7,1] = 1

    board[8,7] = -1
    board[8,8] = 1

    import os
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_model.pth")
    predictor = HexModelInference(model_path)
    policy, value = predictor.predict(board, current_player=-1)
    print("Predicted Policy:", policy)
    print("Predicted Value:", value)

if __name__ == "__main__":
    main()


