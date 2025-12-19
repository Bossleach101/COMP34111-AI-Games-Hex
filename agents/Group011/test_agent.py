import sys
import os
sys.path.append(os.getcwd())

from agents.Group11.Group11Agent import Group11Agent
from src.Colour import Colour
from src.Board import Board
from src.Move import Move

def test_agent():
    print("Initializing Agent...")
    agent = Group11Agent(Colour.RED)
    print("Agent Initialized.")
    
    board = Board() # src.Board
    
    print("Making first move...")
    move = agent.make_move(1, board, None)
    print(f"Agent made move: {move.x}, {move.y}")
    
    print("Updating with opponent move...")
    opp_move = Move(0, 0)
    if move.x == 0 and move.y == 0:
        opp_move = Move(0, 1)
        
    move2 = agent.make_move(2, board, opp_move)
    print(f"Agent made second move: {move2.x}, {move2.y}")

if __name__ == "__main__":
    test_agent()