import numpy as np
from scipy.ndimage import label
from scipy.ndimage import binary_dilation

class Board:
    def __init__(self, board, first_to_play=1):
        # Ensure board is a numpy array
        if not isinstance(board, np.ndarray):
            self.board = np.array(board, dtype=int)
        else:
            self.board = board
            
        self.size = self.board.shape[0]
        self.first_to_play = first_to_play
        
        # Hex neighborhood structure for scipy.ndimage.label
        # Corresponds to:
        # (-1, 0), (-1, 1)
        # (0, -1), (0, 1)
        # (1, -1), (1, 0)
        self.hex_structure = np.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ], dtype=int)
    
    def copy(self):
        """Create a deep copy of the board"""
        return Board(self.board.copy(), self.first_to_play)
    
    def eval(self):
        """
        Evaluate the board state.
        Returns: 1 if Red (1) wins, -1 if Blue (-1) wins, 0 otherwise
        """
        winner = self.get_winner()
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        return 0
    
    def get_valid_moves(self, use_heuristic=False, max_distance=2):
        """
        Return list of valid moves.
        """
        # Get all empty cells
        empty_indices = np.argwhere(self.board == 0)
        moves = [tuple(x) for x in empty_indices]

        if not use_heuristic or not moves:
            return moves

        # Check if board is empty (no pieces)
        if np.all(self.board == 0):
            return moves

        # Heuristic: Filter moves near existing pieces
        # Create a mask of occupied cells
        occupied = (self.board != 0)
        
        # Dilate the occupied mask to find neighbors within max_distance
        # We can use binary_dilation from scipy.ndimage
                
        # Create a structure for dilation based on max_distance
        # For max_distance=1, it's the hex_structure.
        # For max_distance=2, we iterate dilation twice or precompute structure.
        # Iterating is easier.
        
        mask = occupied
        for _ in range(max_distance):
            mask = binary_dilation(mask, structure=self.hex_structure)
            
        # Valid moves are empty cells that are in the dilated mask
        # (i.e., near occupied cells)
        # But we must exclude occupied cells themselves (already handled by checking empty_indices)
        
        # Filter moves
        filtered_moves = []
        for r, c in moves:
            if mask[r, c]:
                filtered_moves.append((r, c))
                
        if filtered_moves:
            return filtered_moves
            
        return moves

    def make_move(self, move, player):
        """Make a move on the board"""
        self.board[move] = player
    
    def is_terminal(self):
        """Check if game is over"""
        return self.get_winner() is not None or not np.any(self.board == 0)
        
    def get_winner(self):
        """
        Return winner for Hex game.
        Player 1 (Red) connects top-bottom (rows 0 to size-1)
        Player -1 (Blue) connects left-right (cols 0 to size-1)
        Returns: 1, -1, or None
        """
        # Check Red (1) - Vertical
        red_stones = (self.board == 1)
        if np.any(red_stones[0, :]) and np.any(red_stones[-1, :]):
            labeled, n_features = label(red_stones, structure=self.hex_structure)
            # Check if any label appears in both top and bottom rows
            top_labels = np.unique(labeled[0, :])
            bottom_labels = np.unique(labeled[-1, :])
            # 0 is background, ignore it
            top_labels = top_labels[top_labels != 0]
            bottom_labels = bottom_labels[bottom_labels != 0]
            
            if np.intersect1d(top_labels, bottom_labels).size > 0:
                return 1

        # Check Blue (-1) - Horizontal
        blue_stones = (self.board == -1)
        if np.any(blue_stones[:, 0]) and np.any(blue_stones[:, -1]):
            labeled, n_features = label(blue_stones, structure=self.hex_structure)
            # Check if any label appears in both left and right columns
            left_labels = np.unique(labeled[:, 0])
            right_labels = np.unique(labeled[:, -1])
            # 0 is background
            left_labels = left_labels[left_labels != 0]
            right_labels = right_labels[right_labels != 0]
            
            if np.intersect1d(left_labels, right_labels).size > 0:
                return -1

        return None

    def get_current_player(self):
        """
        Determine whose turn it is.
        """
        count_1 = np.sum(self.board == 1)
        count_neg_1 = np.sum(self.board == -1)

        if self.first_to_play == 1:
            if count_1 > count_neg_1:
                return -1
            else:
                return 1
        else:
            if count_1 > count_neg_1:
                return 1
            else:
                return -1

    def find_winning_move(self, player):
        """
        Check if there is any move that immediately wins for the given player.
        Returns the move (r, c) or None.
        """
        # Only check moves near existing pieces (heuristic) as a winning move must connect things.
        # If board is empty or sparse, no immediate win is possible anyway.
        valid_moves = self.get_valid_moves(use_heuristic=True, max_distance=2)
        
        for move in valid_moves:
            # Try move
            self.board[move] = player
            is_win = (self.get_winner() == player)
            # Undo move
            self.board[move] = 0
            
            if is_win:
                return move
        return None

