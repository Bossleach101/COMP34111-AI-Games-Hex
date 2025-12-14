import numpy as np

class HexPlanes:
    """
    Class to generate various feature planes for Hex game states.
    """
    # Constants for the board dimensions
    _BOARD_SIZE = 11

    # The 6 Bridge Patterns in Axial Coordinates (q, r)
    # Format: (Target_Offset, [Intermediate_1, Intermediate_2])
    _BRIDGE_OFFSETS = [
        ((+1, -2), [(0, -1), (+1, -1)]),
        ((+2, -1), [(+1, -1), (+1, 0)]),
        ((+1, +1), [(+1, 0), (0, +1)]),
        ((-1, +2), [(0, +1), (-1, +1)]),
        ((-2, +1), [(-1, +1), (-1, 0)]),
        ((-1, -1), [(-1, 0), (0, -1)])
    ]

    @staticmethod
    def get_blue_red_plane(board_state):
        blue_plane = (board_state == -1).astype(int)
        red_plane = (board_state == 1).astype(int)
        return blue_plane, red_plane
    
    @staticmethod
    def get_turn_plane(current_player):
        if current_player == 1:
            return np.ones((HexPlanes._BOARD_SIZE, HexPlanes._BOARD_SIZE), dtype=np.float32)
        else:
            return -1*np.ones((HexPlanes._BOARD_SIZE, HexPlanes._BOARD_SIZE), dtype=np.float32)

    @staticmethod
    # Utility functions for Hex game agents
    def get_bridge_plane(board_state, player_color):
        """
        Generates the Bridge Plane for a specific player.
        
        Args:
            board_state: 11x11 numpy array (0=Empty, 1=Black, -1=White)
                        (Or however you encode your raw board)
            player_color: The value representing the current player (e.g., 1 or -1)
        
        Returns:
            11x11 numpy array (1 where a bridge connection exists, 0 otherwise)
        """
        

        bridge_plane = np.zeros((HexPlanes._BOARD_SIZE, HexPlanes._BOARD_SIZE), dtype=np.float32)
        
        # Get all coordinates where the player currently has stones
        # distinct_stones is a list of (r, q) or (x, y) depending on your setup
        # Assuming board_state is indexed [r][q] (row, col)
        rows, cols = np.where(board_state == player_color)
        my_stones = list(zip(rows, cols))

        for r, q in my_stones:
            # Check all 6 possible bridge directions from this stone
            for target_diff, intermediates in HexPlanes._BRIDGE_OFFSETS:
                
                # 1. Identify the Target Stone location
                target_r = r + target_diff[1] # Note: Ensure r/q mapping matches your grid
                target_q = q + target_diff[0] # standard axial is often (q, r)
                
                # Boundary Check: Is target on the board?
                if not (0 <= target_r < HexPlanes._BOARD_SIZE and 0 <= target_q < HexPlanes._BOARD_SIZE):
                    continue
                    
                # Friend Check: Is there a friendly stone at the target?
                # Note: We only need to check "forward" directions to avoid double counting,
                # but checking all is safer and easier to debug.
                if board_state[target_r, target_q] == player_color:
                    
                    # 2. Check the Intermediates
                    valid_bridge = True
                    int_coords = []
                    
                    for mid_diff in intermediates:
                        mid_r = r + mid_diff[1]
                        mid_q = q + mid_diff[0]
                        
                        # If intermediate is off board, it's not a valid bridge
                        if not (0 <= mid_r < HexPlanes._BOARD_SIZE and 0 <= mid_q < HexPlanes._BOARD_SIZE):
                            valid_bridge = False
                            break
                        
                        # If intermediate is NOT empty, the bridge is broken
                        if board_state[mid_r, mid_q] != 0:
                            valid_bridge = False
                            break
                            
                        int_coords.append((mid_r, mid_q))
                    
                    # 3. Mark the Plane
                    if valid_bridge:
                        for (mr, mq) in int_coords:
                            bridge_plane[mr, mq] = 1.0

        return bridge_plane
    
    @staticmethod
    def get_all_feature_planes(board_state, current_player):
        blue_plane, red_plane = HexPlanes.get_blue_red_plane(board_state)
        turn_plane = HexPlanes.get_turn_plane(current_player)
        bridge_plane = HexPlanes.get_bridge_plane(board_state, current_player)
        
        feature_planes = np.stack([blue_plane, red_plane, turn_plane, bridge_plane], axis=0)
        return feature_planes


if __name__ == "__main__":
    
    board = np.zeros((11, 11), dtype=int)
    board[0, 0] = -1 
    board[0, 1] = 0
    board[1, 1] = -1 
    board[2, 0] = -1
    board[5, 5] = 1
    board[6, 5] = 1
    board[5, 6] = 1 
    print(HexPlanes.get_all_feature_planes(board, -1))