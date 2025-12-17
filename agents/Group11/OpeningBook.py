from __future__ import annotations

from collections.abc import Iterable

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class OpeningBook:
    """
    Opening database for 11x11 Hex.


    The book only intervenes for the first ten moves (five moves each side)
    and falls back to MCTS afterwards.
    """

    def __init__(self, board_size: int = 11):
        self.board_size = board_size
        self.centre = board_size // 2
        # Star points: centre plus its immediate neighbours
        # opening charts emphasise for 11x11 symmetry.
        self.star_points = [
            (self.centre, self.centre),
            (self.centre - 1, self.centre),
            (self.centre + 1, self.centre),
            (self.centre, self.centre - 1),
            (self.centre, self.centre + 1),
            (self.centre - 1, self.centre + 1),
            (self.centre + 1, self.centre - 1),
        ]

    def lookup(self, colour: Colour, board: Board, opp_move: Move | None) -> Move | None:
        if board.size != self.board_size:
            return None

        occupied = self._count_stones(board)

        if occupied >= 10:
            return None

        if occupied == 0 and colour == Colour.RED:
            return self._open_as_first(board)

        if occupied == 1 and colour == Colour.BLUE:
            return self._respond_as_second(board, opp_move)

        if occupied == 2 and colour == Colour.RED:
            return self._reinforce_as_first(board)

        if occupied == 3 and colour == Colour.BLUE:
            return self._develop_as_second(board)

        if occupied == 4 and colour == Colour.RED:
            return self._advance_as_first(board)

        if occupied == 5 and colour == Colour.BLUE:
            return self._consolidate_as_second(board)

        if occupied == 6 and colour == Colour.RED:
            return self._extend_as_first(board)

        if occupied == 7 and colour == Colour.BLUE:
            return self._widen_as_second(board)

        if occupied == 8 and colour == Colour.RED:
            return self._link_as_first(board)

        if occupied == 9 and colour == Colour.BLUE:
            return self._stabilise_as_second(board)

        return None

    # --- Internal helpers -------------------------------------------------

    def _count_stones(self, board: Board) -> int:
        return sum(1 for row in board.tiles for tile in row if tile.colour is not None)

    def _is_empty(self, board: Board, move: tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour is None

    def _first_available(self, board: Board, moves: Iterable[tuple[int, int]]) -> Move | None:
        for x, y in moves:
            if self._is_empty(board, (x, y)):
                return Move(x, y)
        return None

    def _open_as_first(self, board: Board) -> Move | None:
        # Play the central point or its nearest star neighbours

        return self._first_available(board, self.star_points)

    def _respond_as_second(self, board: Board, opp_move: Move | None) -> Move | None:
        """
        Second player's (Blue) first move.

        Strong central openings should be swapped; otherwise, take the centre if
        it's still free or mirror the opponent towards our horizontal plan.
        """

        strong_openings = set(self.star_points)

        if opp_move and (opp_move.x, opp_move.y) in strong_openings:
            # Pie rule: centre/star openings are worth swapping
            return Move(-1, -1)

        # If centre is free, claim it; otherwise mirror the opponent across the board.
        if self._is_empty(board, (self.centre, self.centre)):
            return Move(self.centre, self.centre)

        if opp_move:
            mirrored = (self.board_size - 1 - opp_move.x, self.board_size - 1 - opp_move.y)
            if self._is_empty(board, mirrored):
                return Move(*mirrored)

        # Fall back to a horizontal anchor near the middle line.
        horizontal_pref = [
            (self.centre, self.centre - 1),
            (self.centre, self.centre + 1),
            (self.centre, self.centre - 2),
            (self.centre, self.centre + 2),
        ]

        return self._first_available(board, horizontal_pref)

    def _reinforce_as_first(self, board: Board) -> Move | None:
        # Red's second move (move 3)

        vertical_rungs = [
            (self.centre - 1, self.centre),
            (self.centre + 1, self.centre),
            (self.centre - 2, self.centre),
            (self.centre + 2, self.centre),
        ]

        diagonal_support = [
            (self.centre - 1, self.centre - 1),
            (self.centre + 1, self.centre + 1),
            (self.centre, self.centre - 1),
            (self.centre, self.centre + 1),
        ]

        return (
            self._first_available(board, vertical_rungs)
            or self._first_available(board, diagonal_support)
            or self._first_available(board, self.star_points)
        )

    def _develop_as_second(self, board: Board) -> Move | None:
        # Blue's second move (move 4)

        lateral_steps = [
            (self.centre, self.centre - 1),
            (self.centre, self.centre + 1),
            (self.centre, self.centre - 2),
            (self.centre, self.centre + 2),
            (self.centre - 1, self.centre),
            (self.centre + 1, self.centre),
        ]

        return self._first_available(board, lateral_steps)

    def _advance_as_first(self, board: Board) -> Move | None:
        # Red's third move (move 5)
        forward_links = [
            (self.centre - 2, self.centre - 1),
            (self.centre + 2, self.centre + 1),
            (self.centre - 2, self.centre + 1),
            (self.centre + 2, self.centre - 1),
            (self.centre - 3, self.centre),
            (self.centre + 3, self.centre),
            (self.centre - 1, self.centre + 1),
            (self.centre + 1, self.centre - 1),
        ]

        return self._first_available(board, forward_links)

    def _consolidate_as_second(self, board: Board) -> Move | None:
        # Blue's third move (move 6)

        horizontal_push = [
            (self.centre, self.centre - 2),
            (self.centre, self.centre + 2),
            (self.centre, self.centre - 3),
            (self.centre, self.centre + 3),
            (self.centre - 1, self.centre - 1),
            (self.centre + 1, self.centre + 1),
            (self.centre - 1, self.centre),
            (self.centre + 1, self.centre),
        ]

        return self._first_available(board, horizontal_push)

    def _extend_as_first(self, board: Board) -> Move | None:
        # Red's fourth move (move 7)

        stretch = [
            (self.centre - 3, self.centre - 1),
            (self.centre + 3, self.centre + 1),
            (self.centre - 4, self.centre),
            (self.centre + 4, self.centre),
            (self.centre - 2, self.centre + 2),
            (self.centre + 2, self.centre - 2),
            (self.centre - 1, self.centre + 2),
            (self.centre + 1, self.centre - 2),
        ]

        return self._first_available(board, stretch)

    def _widen_as_second(self, board: Board) -> Move | None:
        # Blue's fourth move (move 8)


        widen = [
            (self.centre, self.centre - 4),
            (self.centre, self.centre + 4),
            (self.centre - 2, self.centre - 2),
            (self.centre + 2, self.centre + 2),
            (self.centre - 1, self.centre + 2),
            (self.centre + 1, self.centre - 2),
            (self.centre - 2, self.centre),
            (self.centre + 2, self.centre),
        ]

        return self._first_available(board, widen)

    def _link_as_first(self, board: Board) -> Move | None:
        # Red's fifth move (move 9)

        links = [
            (self.centre - 2, self.centre - 2),
            (self.centre + 2, self.centre + 2),
            (self.centre - 3, self.centre + 1),
            (self.centre + 3, self.centre - 1),
            (self.centre - 1, self.centre + 3),
            (self.centre + 1, self.centre - 3),
            (self.centre - 4, self.centre),
            (self.centre + 4, self.centre),
        ]

        return self._first_available(board, links)

    def _stabilise_as_second(self, board: Board) -> Move | None:
        # Blue's fifth move (move 10)

        stabilise = [
            (self.centre, self.centre - 5),
            (self.centre, self.centre + 5),
            (self.centre - 3, self.centre + 1),
            (self.centre + 3, self.centre - 1),
            (self.centre - 1, self.centre + 3),
            (self.centre + 1, self.centre - 3),
            (self.centre - 2, self.centre + 2),
            (self.centre + 2, self.centre - 2),
        ]

        return self._first_available(board, stabilise)
