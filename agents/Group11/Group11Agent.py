from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import agents.Group11.mcts.mcts as mcts
from agents.Group11.BoardEvaluator import HexModelInference
from agents.Group11.OpeningBook import OpeningBook
import os

class Group11Agent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        
        # Map Colour to MCTS Player ID (1=Red, 2=Blue)
        self.player_id = 1 if self.colour == Colour.RED else 2
        self.opp_player_id = 2 if self.player_id == 1 else 1

        # Initialize Predictor
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_model.pth")
        self.predictor = HexModelInference(model_path)

        # Initialize MCTS
        # Assuming Red always plays first in the game logic
        self.mcts_agent = mcts.MCTS(predictor=self.predictor, first_to_play=1, exploration_constant=0.5)

        # Small deterministic opening book for the first few plies
        self.opening_book = OpeningBook(board_size=11)
        self.pending_swap_reset = False

    def _refresh_player_mapping(self):
        """Refresh cached IDs in case our colour changes after a swap."""

        self.player_id = 1 if self.colour == Colour.RED else 2
        self.opp_player_id = 2 if self.player_id == 1 else 1

    def _board_to_array(self, board: Board) -> list[list[int]]:
        """Convert framework board into the integer layout used by our MCTS."""

        translated_board: list[list[int]] = []
        for row in board.tiles:
            new_row = []
            for tile in row:
                if tile.colour == Colour.RED:
                    new_row.append(1)
                elif tile.colour == Colour.BLUE:
                    new_row.append(2)
                else:
                    new_row.append(0)
            translated_board.append(new_row)
        return translated_board

    def _rebuild_tree_from_board(self, board: Board):
        """Hard reset the search tree to the current board position."""

        board_state = self._board_to_array(board)
        self.mcts_agent.reset_to_board(board_state, first_to_play=1)

    def _needs_reset(self, board: Board) -> bool:
        current_board = self._board_to_array(board)
        root_board = self.mcts_agent.root.state.board

        if len(current_board) != len(root_board):
            return True

        # Check how many stones differ between our tree state and the live board
        differences = 0
        for i in range(len(current_board)):
            for j in range(len(current_board)):
                if current_board[i][j] != root_board[i][j]:
                    differences += 1
        # A difference of one stone is expected before we apply the opponent's move.
        return differences > 1

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        self._refresh_player_mapping()

        # Re-sync tree if we swapped last turn or if the board diverged.
        if self.pending_swap_reset or (opp_move and opp_move.is_swap()) or self._needs_reset(board):
            self._rebuild_tree_from_board(board)
            self.pending_swap_reset = False
        elif opp_move and not opp_move.is_swap():
            # Update tree with opponent's move
            self.mcts_agent.update_root((opp_move.x, opp_move.y), self.opp_player_id)

        # Try opening book before invoking MCTS
        opening_move = self.opening_book.lookup(self.colour, board, opp_move)
        if opening_move is not None:
            if opening_move.is_swap():
                self.pending_swap_reset = True
                return opening_move

            self.mcts_agent.update_root((opening_move.x, opening_move.y), self.player_id)
            return opening_move

        # Run MCTS
        # Adjust iterations based on time constraints if needed
        best_move = self.mcts_agent.search(iterations=200)

        # Update tree with our move
        self.mcts_agent.update_root(best_move, self.player_id)

        return Move(best_move[0], best_move[1])
