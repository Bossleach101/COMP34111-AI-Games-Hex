# Save the MCTS code as 'mcts_hex.py', then use this to play:

from mcts.mcts import MCTS
from mcts.board import Board
from mcts.node import Node

def print_board(board):
    """Pretty print the Hex board with borders showing player sides"""
    size = board.size

    # Top border - Player 1 (X)
    print("1 " * (size + 2))

    for i in range(size):
        # Left border - Player 2 (O)
        print("2 ", end="")

        # Board row with proper indentation
        for j in range(size):
            cell = board.board[i][j]
            if cell == 0:
                print(". ", end="")
            elif cell == 1:
                print("X ", end="")
            else:
                print("O ", end="")

        # Right border - Player 2 (O)
        print("2")

    # Bottom border - Player 1 (X)
    print( "1 " * (size + 2))

def play_game():
    """Play a game of Hex: Human vs AI"""
    mcts = MCTS()
    current_player = 1  # 1 = Human (X), 2 = AI (O)
    
    print("Hex Game: Connect top-bottom with X")
    print("AI will connect left-right with O")
    print()
    
    while True:
        print(f"current player {current_player}")
        print_board(mcts.root.state)
        print()
        
        # Check if game is over
        winner = mcts.root.state.get_winner()
        if winner:
            print(f"Player {winner} wins!")
            break
        
        if not mcts.root.state.get_valid_moves():
            print("Draw!")
            break
        
        if current_player == 1:
            # Human turn
            print("Your turn (X). Enter move as 'row col' (0-indexed):")
            try:
                row, col = map(int, input().split())
                if (row, col) not in mcts.root.state.get_valid_moves():
                    print("Invalid move! Try again.")
                    continue
                
                move = (row, col)
            except:
                print("Invalid input! Enter as: row col")
                continue
        else:
            # AI turn
            print("AI thinking...")
            move = mcts.search(iterations=1000)
            print(f"AI plays: {move}")
        
        # Make move and update root
        mcts.update_root(move, current_player)
        
        # Switch player
        current_player = 3 - current_player
        print()

def ai_vs_ai():
    """Watch two AIs play against each other"""
    mcts = MCTS()
    current_player = 1
    move_count = 0
    
    print("AI vs AI Game Starting...")
    print()
    
    while True:
        print(f"Move {move_count + 1}:")
        print_board(mcts.root.state)
        print()
        
        winner = mcts.root.state.get_winner()
        if winner:
            print(f"Player {winner} wins!")
            break
        
        if not mcts.root.state.get_valid_moves():
            print("Draw!")
            break
        
        # AI makes move
        print(f"Player {current_player} thinking...")
        move = mcts.search(iterations=500)
        print(f"Player {current_player} plays: {move}")

        mcts.update_root(move, current_player)
        
        current_player = 3 - current_player
        move_count += 1
        print()

if __name__ == "__main__":
    print("1. Play against AI")
    print("2. Watch AI vs AI")
    choice = input("Choose (1 or 2): ")
    
    if choice == "1":
        play_game()
    else:
        ai_vs_ai()