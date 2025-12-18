from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import agents.Group11.mcts.mcts as mcts
from agents.Group11.BoardEvaluator import HexModelInference
from agents.Group11.mcts.board import Board as MCTSBoard
from agents.Group11.mcts.node import Node as MCTSNode
import numpy as np
import os

class Group11Agent(AgentBase):
    def __init__(self, colour: Colour, **mcts_kwargs):
        super().__init__(colour)

        # Map Colour to MCTS Player ID (1=Red, -1=Blue)
        self.player_id = 1 if self.colour == Colour.RED else -1
        self.opp_player_id = -1 if self.player_id == 1 else 1
        
        # Initialize Predictor
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_model.pth")
        self.predictor = HexModelInference(model_path)
        
        # Default MCTS params
        mcts_params = {
            'predictor': self.predictor,
            'first_to_play': 1,
            'exploration_constant': 1.41, # Default to 1.41
            'selection_policy': "puct"
        }
        
        # Extract iterations if present, otherwise default to 500
        self.iterations = mcts_kwargs.pop('iterations', 500)
        
        # Override with kwargs
        mcts_params.update(mcts_kwargs)
        self._mcts_params = mcts_params

        # Initialize MCTS
        self.mcts_agent = mcts.MCTS(**mcts_params)
        self._pending_swap_sync = False
        self._last_colour = self.colour
        self._swapped_last_turn = False


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        self._refresh_player_ids()

        # Re-sync the search tree when our colour changes (e.g., after a swap)
        if self._last_colour != self.colour or self._pending_swap_sync:
            self._sync_tree_with_board(board)
            self._pending_swap_sync = False
            self._last_colour = self.colour

        if opp_move and not opp_move.is_swap():
            # Update tree with opponent's move
            self.mcts_agent.update_root((opp_move.x, opp_move.y), self.opp_player_id)

        # Opening book and swap consideration
        swap_move = self._consider_swap(turn, board, opp_move)
        if swap_move:
            # Tree will be re-synchronised once our colour updates
            self._pending_swap_sync = True
            self._swapped_last_turn = True
            return swap_move

        if self._swapped_last_turn and self.colour == Colour.RED:
            post_swap_move = self._post_swap_red_follow_up(board)
            if post_swap_move:
                self._swapped_last_turn = False
                self.mcts_agent.update_root((post_swap_move.x, post_swap_move.y), self.player_id)
                return post_swap_move
            self._swapped_last_turn = False

        opening_move = self._opening_book(turn, board, opp_move)
        if opening_move:
            self.mcts_agent.update_root((opening_move.x, opening_move.y), self.player_id)
            return opening_move
        
        # 1. Check for immediate win
        winning_move = self.mcts_agent.root.state.find_winning_move(self.player_id)
        if winning_move:
             # print(f"DEBUG: Found winning move at {winning_move}")
             self.mcts_agent.update_root(winning_move, self.player_id)
             return Move(winning_move[0], winning_move[1])

        # 2. Check for immediate loss (opponent win) and block it
        blocking_move = self.mcts_agent.root.state.find_winning_move(self.opp_player_id)
        if blocking_move:
             # print(f"DEBUG: Found blocking move at {blocking_move}")
             self.mcts_agent.update_root(blocking_move, self.player_id)
             return Move(blocking_move[0], blocking_move[1])

        # Run MCTS
        # Adjust iterations based on time constraints if needed
        best_move = self.mcts_agent.search(iterations=self.iterations)
        
        # Update tree with our move
        self.mcts_agent.update_root(best_move, self.player_id)

        return Move(best_move[0], best_move[1])

    def _refresh_player_ids(self):
        self.player_id = 1 if self.colour == Colour.RED else -1
        self.opp_player_id = -self.player_id

    def _sync_tree_with_board(self, board: Board):
        board_array = self._board_to_array(board)
        new_root = MCTSNode(MCTSBoard(board_array, self._mcts_params.get('first_to_play', 1)))
        self.mcts_agent = mcts.MCTS(**self._mcts_params)
        self.mcts_agent.root = new_root

    def _post_swap_red_follow_up(self, board: Board) -> Move | None:
        size = board.size
        red_stones = [
            (x, y)
            for x in range(size)
            for y in range(size)
            if board.tiles[x][y].colour == Colour.RED
        ]

        if not red_stones:
            return None

        # Use the first Red stone found (should be the inherited opener)
        x, y = red_stones[0]
        direction = 1 if x <= size // 2 else -1
        candidates = [
            (x + direction, y),
            (x - direction, y),
        ]

        for cx, cy in candidates:
            if 0 <= cx < size and 0 <= cy < size and board.tiles[cx][cy].colour is None:
                return Move(cx, cy)

        return None

    def _board_to_array(self, board: Board) -> np.ndarray:
        array = np.zeros((board.size, board.size), dtype=int)
        for x in range(board.size):
            for y in range(board.size):
                tile_colour = board.tiles[x][y].colour
                if tile_colour == Colour.RED:
                    array[x, y] = 1
                elif tile_colour == Colour.BLUE:
                    array[x, y] = -1
        return array

    def _consider_swap(self, turn: int, board: Board, opp_move: Move | None) -> Move | None:
        if self.colour != Colour.BLUE or turn != 2 or opp_move is None or opp_move.is_swap():
            return None

        if self._should_swap(board.size, opp_move.x, opp_move.y):
            return Move(-1, -1)
        return None

    def _should_swap(self, size: int, x: int, y: int) -> bool:
        center = size // 2
        # If Red opens anywhere in the central 5x5, take the swap as Blue
        if size >= 5:
            if (center - 2) <= x <= (center + 2) and (center - 2) <= y <= (center + 2):
                return True

        distance = max(abs(x - center), abs(y - center))
        edge_margin = min(x, y, size - 1 - x, size - 1 - y)

        # Swap if the first move is strong (near centre and not hugging an edge)
        if distance <= 1:
            return True
        if distance == 2 and edge_margin >= 2:
            return True
        return False

    def _opening_book(self, turn: int, board: Board, opp_move: Move | None) -> Move | None:
        if board.size != 11:
            return None

        center = board.size // 2

        if self.colour == Colour.RED:
            if turn == 1:
                preferred = (1, board.size - 3)
                # Fixed anti-swap opener for Red on 11x11
                return Move(preferred[0], preferred[1])

            if turn == 3:
                # Preferred extension for the 1,8 opener: step down toward Blue edge
                candidates = [
                    (2, board.size - 4),  # (2,7) on 11x11
                    (2, board.size - 3),  # (2,8) on 11x11
                ]
                for x, y in candidates:
                    if board.tiles[x][y].colour is None:
                        return Move(x, y)

        if self.colour == Colour.BLUE and turn == 2 and opp_move is not None and not opp_move.is_swap():
            if board.tiles[center][center].colour is None:
                return Move(center, center)

            mirrored = (board.size - 1 - opp_move.x, board.size - 1 - opp_move.y)
            if board.tiles[mirrored[0]][mirrored[1]].colour is None:
                return Move(mirrored[0], mirrored[1])

            near_center = [
                (center, center - 1),
                (center, center + 1),
                (center - 1, center + 1),
                (center + 1, center - 1),
            ]
            for x, y in near_center:
                if board.tiles[x][y].colour is None:
                    return Move(x, y)

        if self.colour == Colour.BLUE and turn == 4:
            progression_move = self._blue_progression_move(board)
            if progression_move:
                return progression_move

        return None

    def _blue_progression_move(self, board: Board) -> Move | None:
        size = board.size
        blue_stones = [
            (x, y)
            for x in range(size)
            for y in range(size)
            if board.tiles[x][y].colour == Colour.BLUE
        ]

        if not blue_stones:
            return None

        # Prefer moves that extend horizontally toward the nearest edge
        blue_stones.sort(key=lambda pos: pos[1])
        for x, y in blue_stones:
            direction = 1 if y <= size // 2 else -1
            candidates = [
                (x, y + direction),
                (x - 1, y + direction),
                (x + 1, y + direction),
            ]

            for cx, cy in candidates:
                if 0 <= cx < size and 0 <= cy < size and board.tiles[cx][cy].colour is None:
                    return Move(cx, cy)

        return None
