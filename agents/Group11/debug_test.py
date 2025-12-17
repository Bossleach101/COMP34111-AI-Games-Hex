from mcts.node import Node
from mcts.board import Board
from mcts.mcts import MCTS

board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

state = Board(board)
mcts = MCTS(1)

# Run search
mcts.search(iterations=1000)

# Analyze tree structure
print(f"Root visits: {mcts.root.visits}")
print(f"Root children: {len(mcts.root.children)}")
print()

# Look at visit distribution of level-1 children
child_visits = sorted([child.visits for child in mcts.root.children], reverse=True)
print(f"Top 10 most visited level-1 children: {child_visits[:10]}")
print(f"Average visits per level-1 child: {sum(child_visits) / len(child_visits):.2f}")
print(f"Max visits on any level-1 child: {max(child_visits)}")
print()

# Find the most visited child and check its expansion
most_visited_child = max(mcts.root.children, key=lambda c: c.visits)
print(f"Most visited child: move={most_visited_child.move}, visits={most_visited_child.visits}")
print(f"Most visited child has {len(most_visited_child.children)} children")
print(f"Most visited child is_fully_expanded: {most_visited_child.is_fully_expanded()}")
print()

# Calculate tree depth
def get_max_depth(node, current_depth=0):
    if not node.children:
        return current_depth
    return max(get_max_depth(child, current_depth + 1) for child in node.children)

print(f"Tree max depth: {get_max_depth(mcts.root)}")
