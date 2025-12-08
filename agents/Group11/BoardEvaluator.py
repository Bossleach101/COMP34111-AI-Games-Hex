from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import torch
import numpy as np

class BoardEvaluator:
    
    _boaard_size = 11
    
    
    # Inputs Board and the colour whose turn it is
    @staticmethod
    def get_board_evaluation(board: Board, colour: Colour) -> float:
        
        red = [[0 for _ in range(BoardEvaluator._boaard_size)] for _ in range(BoardEvaluator._boaard_size)]
        blue = [[0 for _ in range(BoardEvaluator._boaard_size)] for _ in range(BoardEvaluator._boaard_size)]
        
        for i in range(board.size):
            for j in range(board.size):
                tile = board.tiles[i][j]
                if tile.colour == Colour.RED:
                    red[i][j] = 1
                elif tile.colour == Colour.BLUE:
                    blue[i][j] = 1
                    
                    
        if colour == Colour.RED:
            colour = [[1 for _ in range(BoardEvaluator._boaard_size)] for _ in range(BoardEvaluator._boaard_size)]
        else:
            colour = [[0 for _ in range(BoardEvaluator._boaard_size)] for _ in range(BoardEvaluator._boaard_size)]    
        
        