from agents.Group011.mcts.board import Board
from agents.Group011.mcts.node import Node
import random
import numpy as np
from src.Colour import Colour

"""
MCTS with switchable selection policies (PUCT vs UCT)

Usage examples:
    # Use PUCT (default) - leverages neural network policy priors
    mcts = MCTS(predictor, exploration_constant=0.5, selection_policy="puct")

    # Use classic UCT - ignores policy priors, uses only visit counts
    mcts = MCTS(predictor, exploration_constant=1.41, selection_policy="uct")
"""

class MCTS:

    def __init__(self, colour, predictor, first_to_play=1, exploration_constant=0.5, selection_policy="puct"):
        """
        Initialize MCTS

        Args:
            predictor: Neural network predictor for policy and value
            first_to_play: Player who plays first (1 or 2)
            exploration_constant: Exploration parameter (c_puct for PUCT, c for UCT)
            selection_policy: "puct" or "uct" - selection policy to use
        """
        init_state = np.zeros((11, 11), dtype=int)
        self.root = Node(Board(init_state, first_to_play))
        self.exploration_constant = exploration_constant
        self.predictor = predictor
        self.selection_policy = selection_policy
        self.colour = colour
    
    def search(self, iterations=1000):
        """Run MCTS search for given number of iterations"""
        for _ in range(iterations):
            node = self.select(self.root)
            
            value = 0
            if node.is_terminal():
                # Get winner from board: 1 or -1
                winner = node.state.get_winner()
                
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
            
        best_child = max(self.root.children, key=lambda child: child.visits)
        return best_child.move

    def select(self, node):
        """Select a node to expand using tree policy (PUCT or UCT)"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_constant, policy=self.selection_policy)
        return node
    
    def expand_and_evaluate(self, node):
        """
        Evaluate the leaf node using the CNN predictor.
        Expand the node using the policy vector.
        Returns the value (reward) relative to the current player.
        """
        # Board is already numpy array with 1 and -1
        board_np = node.state.board
        
        # Determine CNN current player
        mcts_player = node.state.get_current_player()
        cnn_player = mcts_player
        
        # Predict
        policy, value = self.predictor.predict(board_np, cnn_player)
        # print("q value estimated is ", value)
        
        # Expand node with policy
        node.expand_with_policy(policy)
        
        # Value from CNN is +1 (Red Win) to -1 (Blue Win).
        # We need to return reward relative to the PARENT of the leaf (the player who made the move).
        # The parent of the leaf is the opponent of mcts_player.
        # If mcts_player is 1 (Red), parent is -1 (Blue).
        # If CNN says Red wins (value=1), this is BAD for Blue (-1). Reward should be -1.
        # If CNN says Blue wins (value=-1), this is GOOD for Blue (-1). Reward should be +1.
        # Formula: value * parent_player = value * (-mcts_player)
        
        return value * (-mcts_player)

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