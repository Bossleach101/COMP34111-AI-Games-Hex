import json
import numpy as np

with(open('playhex-games-2025-12-07.json', 'r') as f):
    data = json.load(f)
    
print(f"Total games loaded: {len(data)}")

eleven_games = [game  for game in data if game['boardsize'] == 11
                and game['allowSwap'] == True 
                and (game['outcome'] == 'path')
                and game['movesCount'] > 0]

print(f"Total 11x11 games: {len(eleven_games)}")

print("Mean of game length : "
      + str(sum([game['movesCount'] for game in eleven_games])/len(eleven_games)))

print("Total number of positions: " + str(sum([game['movesCount'] for game in eleven_games])))

def convert_moves_to_position(moves: str) -> np.array:
    posisition = np.zeros((11, 11), dtype=int)
    current_colour = 1  # 1 for RED, -1 for BLUE
    for move in moves.split(' '):
        try:
            if move == "swap-pieces":
                current_colour *= -1
                posisition = -posisition
            elif move == "pass":
                continue
            else:
                col = ord(move[0]) - ord('a')
                row = int(move[1:]) - 1
                posisition[row][col] = current_colour
                current_colour *= -1  # Switch colour
        except Exception as e:
            print(f"Error processing move '{moves}': {e}")
    return posisition

print("Example position from first game:")
example_position = convert_moves_to_position(eleven_games[0]['moves'])
print(example_position)


json_out_batch = []

for game in eleven_games:
    game_moves = ''
    for move in game['moves'].split(' '):
        game_moves += move + ' '
        position = convert_moves_to_position(game_moves.strip())
        outcome = 1 if game['winner'] == 'red' else -1
    
        json_out_batch.append({
            'position': position.tolist(),
            'outcome': outcome
        })

with open('processed_11x11_games.json', 'w') as f:
    json.dump(json_out, f)
    
print("Done")
