from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import agents.Group11.mcts.mcts as mcts
from agents.Group11.BoardEvaluator import HexModelInference
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
        
        # Initialize MCTS
        self.mcts_agent = mcts.MCTS(**mcts_params)


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if opp_move:
            # Update tree with opponent's move
            self.mcts_agent.update_root((opp_move.x, opp_move.y), self.opp_player_id)
        
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