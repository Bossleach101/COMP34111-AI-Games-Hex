from mcts.node import Node
from mcts.board import Board
from mcts.mcts import MCTS
import time

start = time.time()
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
node = Node(state)
mcts = MCTS(1)

mcts.search(iterations=2000)

print(len(mcts.root.children))
j = 0

def dfs(curr = mcts.root):
    global j
    j += 1
    # print("----------")
    # for i in curr.state.board:
    #     print(i)

    if not curr.children: return

    dfs(curr.children[0])

dfs()
print(j)
end = time.time()
print(end-start)