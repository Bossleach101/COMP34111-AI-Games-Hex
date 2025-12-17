from mcts.board import Board
from mcts.node import Node
import random

class MCTS:
    def __init__(self, first_to_play=1, exploration_constant=0.5):
        init_state = [[0]*11 for _ in range(11)]
        self.root = Node(Board(init_state, first_to_play))
        self.exploration_constant = exploration_constant
    
    def search(self, iterations=1000):
        """Run MCTS search for given number of iterations"""
        for _ in range(iterations):
            node = self.select(self.root)
            if not node.is_terminal():
                if not node.is_fully_expanded():
                    # Expand one child at a time (traditional MCTS)
                    expanded_child = node.expand()
                    if expanded_child:
                        node = expanded_child
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        # Return best move using PUCT with c_param=0 (pure exploitation)
        return self.root.best_child(c_param=0).move

    def select(self, node):
        """Select a node to expand using tree policy with PUCT"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child(self.exploration_constant)
        return node
    
    def simulate(self, node):
        """Simulate a random playout from this node using move heuristic"""
        state = node.state.copy()

        while not state.is_terminal():
            # Use heuristic during simulation to speed it up and make it more realistic
            moves = state.get_valid_moves(use_heuristic=True, max_distance=2)
            if not moves:
                break
            move = random.choice(moves)
            # Determine whose turn it is from the current state
            current_player = state.get_current_player()
            state.make_move(move, current_player)

        return state.eval()
    
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
                print(f"DEBUG: Found child with move {move}, using existing child")
                self.root = child
                self.root.parent = None
                return

        # If move not in children, create new root
        print(f"DEBUG: Move {move} not in children, creating new root with player {player}")
        new_state = self.root.state.copy()
        new_state.make_move(move, player)
        self.root = Node(new_state)