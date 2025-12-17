class Board:
    def __init__(self, board, first_to_play = 1):
        self.board = board
        self.size = len(board)
        self.first_to_play = first_to_play
    
    def copy(self):
        """Create a deep copy of the board"""
        return Board([row[:] for row in self.board])
    
    def eval(self):
        """
        Evaluate the board state from Player 2's perspective.
        Returns: 1 if Player 2 wins, -1 if Player 1 wins, 0 otherwise
        """
        winner = self.get_winner()
        if winner == 2:
            return 1  # Player 2 (AI) wins
        elif winner == 1:
            return -1  # Player 1 (opponent) wins
        return 0  # Draw or ongoing
    
    def get_valid_moves(self, use_heuristic=False, max_distance=2):
        """
        Return list of valid moves.

        Args:
            use_heuristic: If True, only return moves near existing pieces
            max_distance: Maximum distance from existing pieces to consider
        """
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    moves.append((i, j))

        # If heuristic enabled and there are pieces on the board, filter moves
        if use_heuristic and moves:
            # Check if board is empty
            has_pieces = any(self.board[i][j] != 0 for i in range(self.size) for j in range(self.size))

            if has_pieces:
                # Only consider moves within max_distance of existing pieces
                filtered_moves = []
                for move in moves:
                    if self._is_near_piece(move, max_distance):
                        filtered_moves.append(move)

                # If filtering resulted in valid moves, use them; otherwise use all moves
                if filtered_moves:
                    return filtered_moves

        return moves

    def _is_near_piece(self, move, max_distance):
        """Check if a move is within max_distance of any existing piece"""
        row, col = move

        # Only check cells within the bounding box of max_distance
        min_row = max(0, row - max_distance)
        max_row = min(self.size - 1, row + max_distance)
        min_col = max(0, col - max_distance)
        max_col = min(self.size - 1, col + max_distance)

        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if self.board[i][j] != 0:
                    # Use hex distance (max of absolute differences in hex coordinates)
                    dist = max(abs(row - i), abs(col - j), abs((row - col) - (i - j)))
                    if dist <= max_distance:
                        return True
        return False
    
    def make_move(self, move, player):
        """Make a move on the board"""
        i, j = move
        self.board[i][j] = player
    
    def is_terminal(self):
        """Check if game is over (someone won or board is full)"""
        return self.get_winner() is not None or len(self.get_valid_moves()) == 0
    
    def _dfs(self, player, current, end_cells, visited):
        """DFS to check if we can reach any end cell from current position"""
        if current in visited:
            return False

        visited.add(current)

        # Check if we reached an end cell
        if current in end_cells:
            return True

        row, col = current

        # Check all 6 neighbors in Hex (adjacent cells)
        neighbors = [
            (row-1, col), (row-1, col+1),  # top-left, top-right
            (row, col-1), (row, col+1),    # left, right
            (row+1, col-1), (row+1, col)   # bottom-left, bottom-right
        ]

        for next_row, next_col in neighbors:
            # Check bounds
            if 0 <= next_row < self.size and 0 <= next_col < self.size:
                # Check if it's the same player's cell
                if self.board[next_row][next_col] == player:
                    if self._dfs(player, (next_row, next_col), end_cells, visited):
                        return True

        return False

    def _has_path(self, player, start_cells, end_cells):
        """Check if there's a path from any start cell to any end cell for player"""
        visited = set()

        # Try starting from each cell in start_cells
        for start in start_cells:
            if self.board[start[0]][start[1]] == player:
                if self._dfs(player, start, end_cells, visited):
                    return True

        return False
        
    def get_winner(self):
        """
        Return winner for Hex game.
        Player 1 connects top-bottom (vertical)
        Player 2 connects left-right (horizontal)
        Returns: 1, 2, or None
        """
        # Check if Player 1 (vertical) has won
        if self._has_path(1,
                         [(0, j) for j in range(self.size)],  # Top row
                         [(self.size-1, j) for j in range(self.size)]):  # Bottom row
            return 1

        # Check if Player 2 (horizontal) has won
        if self._has_path(2,
                         [(i, 0) for i in range(self.size)],  # Left column
                         [(i, self.size-1) for i in range(self.size)]):  # Right column
            return 2

        return None

    def get_current_player(self):
        """
        Determine whose turn it is based on the number of pieces on the board.
        Player 1 goes first, so if there are equal pieces, it's Player 1's turn.
        If Player 1 has more pieces, it's Player 2's turn.
        """
        player1_count = sum(row.count(1) for row in self.board)
        player2_count = sum(row.count(2) for row in self.board)

        # Player 1 goes first
        # If equal pieces or Player 2 has more, it's Player 1's turn
        # If Player 1 has more, it's Player 2's turn
        if self.first_to_play == 1:
            if player1_count > player2_count:
                return 2
            else:
                return 1
        else:
            if player1_count > player2_count:
                return 1
            else:
                return 2

