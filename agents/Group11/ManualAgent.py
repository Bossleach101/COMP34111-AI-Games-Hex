from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class ManualAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        user_input = input(f"Your turn ({self.colour.name}). Enter your move as 'row,col': ")
        row, col = map(int, user_input.split(','))
        return Move(row, col)
