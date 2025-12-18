from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class ManualAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        while True:
            allow_swap = self.colour == Colour.BLUE and turn == 2
            prompt = "Your turn ({}). Enter your move as 'row,col'{}: ".format(
                self.colour.name,
                " or 'swap'" if allow_swap else "",
            )
            user_input = input(prompt).strip().lower()

            if allow_swap and user_input == "swap":
                return Move(-1, -1)

            try:
                row, col = map(int, user_input.split(","))
                return Move(row, col)
            except ValueError:
                print("Invalid input. Please use the format 'row,col'{}.".format(
                    " or 'swap'" if allow_swap else ""
                ))
