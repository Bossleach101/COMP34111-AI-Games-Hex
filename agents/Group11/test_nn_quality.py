"""
Test script to verify if the Neural Network is providing good predictions
"""
import numpy as np
from agents.Group11.BoardEvaluator import HexModelInference
import os

def test_nn_quality():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_model.pth")
    predictor = HexModelInference(model_path)

    print("=" * 60)
    print("Testing Neural Network Quality")
    print("=" * 60)

    # Test 1: Empty board - should have roughly uniform policy
    print("\n1. Empty Board Test:")
    empty_board = np.zeros((11, 11), dtype=int)
    policy, value = predictor.predict(empty_board, current_player=1)
    print(f"   Value prediction: {value:.4f} (should be close to 0)")
    print(f"   Policy entropy: {-np.sum(policy * np.log(policy + 1e-10)):.4f}")
    print(f"   Max policy prob: {np.max(policy):.4f}")
    print(f"   Min policy prob: {np.min(policy):.4f}")

    # Test 2: Near-winning position for Red (player 1)
    print("\n2. Red Near-Win Test:")
    near_win_red = np.zeros((11, 11), dtype=int)
    # Create a path from top to one before bottom
    for i in range(10):
        near_win_red[i, 5] = 1

    policy, value = predictor.predict(near_win_red, current_player=1)
    print(f"   Value prediction: {value:.4f} (should be close to +1)")
    print(f"   Best move predicted: {np.unravel_index(np.argmax(policy), policy.shape)}")
    print(f"   Expected move: (10, 5) to complete connection")
    print(f"   Policy at (10,5): {policy[10, 5]:.4f}")

    # Test 3: Near-winning position for Blue (player -1)
    print("\n3. Blue Near-Win Test:")
    near_win_blue = np.zeros((11, 11), dtype=int)
    # Create a path from left to one before right
    for j in range(10):
        near_win_blue[5, j] = -1

    policy, value = predictor.predict(near_win_blue, current_player=1)
    print(f"   Value prediction: {value:.4f} (should be close to -1)")
    print(f"   Best move predicted: {np.unravel_index(np.argmax(policy), policy.shape)}")
    print(f"   Expected move: (5, 10) to complete connection")
    print(f"   Policy at (5,10): {policy[5, 10]:.4f}")

    # Test 4: Blocking test
    print("\n4. Blocking Test (Red must block Blue):")
    blocking_board = np.zeros((11, 11), dtype=int)
    for j in range(10):
        blocking_board[5, j] = -1
    blocking_board[0, 0] = 1  # Red has one stone

    policy, value = predictor.predict(blocking_board, current_player=1)
    print(f"   Value prediction: {value:.4f} (should be close to -1, Blue winning)")
    print(f"   Best move predicted: {np.unravel_index(np.argmax(policy), policy.shape)}")
    print(f"   Expected move: (5, 10) to block Blue")
    print(f"   Policy at (5,10): {policy[5, 10]:.4f}")

    # Test 5: Check if policy sums to 1
    print("\n5. Policy Normalization Test:")
    print(f"   Policy sum: {np.sum(policy):.6f} (should be 1.0)")

    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    print("If the NN is good:")
    print("  - Value should be close to +1 when Red is winning")
    print("  - Value should be close to -1 when Blue is winning")
    print("  - Policy should put high probability on winning/blocking moves")
    print("  - Policy sum should equal 1.0")
    print("\nIf these tests fail, your NN needs more/better training.")
    print("=" * 60)

if __name__ == "__main__":
    test_nn_quality()
