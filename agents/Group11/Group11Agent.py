from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import agents.Group11.mcts.mcts as mcts
from agents.Group11.BoardEvaluator import HexModelInference
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

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if opp_move:
            # Update tree with opponent's move
            self.mcts_agent.update_root((opp_move.x, opp_move.y), self.opp_player_id)
        
        # Run MCTS
        # Adjust iterations based on time constraints if needed
        best_move = self.mcts_agent.search(iterations=200)
        
        # Update tree with our move
        self.mcts_agent.update_root(best_move, self.player_id)
        
        return Move(best_move[0], best_move[1])