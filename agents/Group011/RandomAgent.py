from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import random



# Agent that plays random valid moves as an example of implementing an agent
# Every other agent we being will need to always win against this agent
class RandomAgent(AgentBase):
    _board_size: int = 11
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        return self.make_random_move(board)
    
    def make_random_move(self, board: Board) -> Move:
        empty_tiles = []
        for i in range(board.size):
            for j in range(board.size):
                t = board.tiles[i][j]
                if t.colour is None:
                    empty_tiles.append((i, j))
        chosen_tile = random.choice(empty_tiles)
        return Move(chosen_tile[0], chosen_tile[1])