from agents.Group11.mcts.board import Board
from agents.Group11.mcts.node import Node
import random
import numpy as np

class MCTS:
    def __init__(self, predictor, first_to_play=1, exploration_constant=0.5):
        init_state = [[0]*11 for _ in range(11)]
        self.root = Node(Board(init_state, first_to_play))
        self.exploration_constant = exploration_constant
        self.predictor = predictor

    def reset_to_board(self, board_state, first_to_play=1):
        """Rebuild the search tree from an explicit board layout."""

        copied_state = [row[:] for row in board_state]
        self.root = Node(Board(copied_state, first_to_play))
    
    def search(self, iterations=1000):
        """Run MCTS search for given number of iterations"""
        for _ in range(iterations):
            node = self.select(self.root)
            
            value = 0
            if node.is_terminal():
                # Get winner from board: 1 or 2
                winner = node.state.get_winner()
                if winner == 1:
                    value = 1 # Red wins
                elif winner == 2:
                    value = -1 # Blue wins
                else:
                    value = 0 # Draw
                
                # Adjust value to be relative to the player who just moved?
                # No, backpropagate flips it.
                # We need to pass the absolute value (Red perspective) and let backpropagate handle it?
                # Wait, backpropagate logic:
                # node.value += reward
                # reward = -reward
                # node = node.parent
                
                # If node is Red (1) to play.
                # If Red wins (value=1).
                # node.value += 1. (Good for Red).
                # Parent (Blue to play). reward = -1.
                # parent.value += -1. (Bad for Blue).
                # This works if we pass +1 for Red win, -1 for Blue win, AND adjust for current player?
                
                # Let's stick to: Reward is relative to the player at the node.
                # If node is Red (1). Red wins. Reward = +1.
                # If node is Blue (2). Blue wins. Reward = +1.
                
                current_player = node.state.get_current_player()
                if winner == current_player:
                    reward = 1
                elif winner is not None:
                    reward = -1
                else:
                    reward = 0
            else:
                # Expand and Evaluate
                reward = self.expand_and_evaluate(node)
            
            self.backpropagate(node, reward)

        # Return best move using PUCT with c_param=0 (pure exploitation)
        return self.root.best_child(c_param=0).move

    def select(self, node):
        """Select a node to expand using tree policy with PUCT"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_constant)
        return node
    
    def expand_and_evaluate(self, node):
        """
        Evaluate the leaf node using the CNN predictor.
        Expand the node using the policy vector.
        Returns the value (reward) relative to the current player.
        """
        # Convert board to numpy array for CNN
        # MCTS Board: 0=Empty, 1=P1(Red), 2=P2(Blue)
        # CNN Board: 0=Empty, 1=Red, -1=Blue
        board_list = node.state.board
        board_np = np.array(board_list, dtype=int)
        board_np[board_np == 2] = -1
        
        # Determine CNN current player
        mcts_player = node.state.get_current_player()
        cnn_player = 1 if mcts_player == 1 else -1
        
        # Predict
        policy, value = self.predictor.predict(board_np, cnn_player)
        
        # Expand node with policy
        node.expand_with_policy(policy)
        
        # Value from CNN is +1 (Red Win) to -1 (Blue Win).
        # We need to return reward relative to mcts_player.
        # If mcts_player is 1 (Red): reward = value * 1 = value.
        # If mcts_player is 2 (Blue): reward = value * -1 = -value.
        # So reward = value * cnn_player
        
        return value * cnn_player

    def backpropagate(self, node, reward):
        """
        Backpropagate the reward up the tree.
        Negate reward at each level since parent and child represent opposing players.
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            reward = -reward  # Flip perspective for parent node
            node = node.parent
    
    def update_root(self, move, player):
        """Update root after a move is made"""
        for child in self.root.children:
            if child.move == move:
                # print(f"DEBUG: Found child with move {move}, using existing child")
                self.root = child
                self.root.parent = None
                return

        # If move not in children, create new root
        # print(f"DEBUG: Move {move} not in children, creating new root with player {player}")
        new_state = self.root.state.copy()
        new_state.make_move(move, player)
        self.root = Node(new_state)
