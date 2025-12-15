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
        N = HexPlanes._BOARD_SIZE
        
        player_mask = (board_state == player_color)
        empty_mask = (board_state == 0)
        
        for (dq, dr), intermediates in HexPlanes._BRIDGE_OFFSETS:
            # Note: BRIDGE_OFFSETS are (q, r) but board is (r, q)
            # So dr corresponds to index 0 change, dq to index 1 change?
            # Wait, usually board[row][col] -> board[r][q].
            # If offsets are (q, r), then dr is row offset, dq is col offset.
            # Let's assume (q, r) -> (col_offset, row_offset)
            
            # Calculate valid range for start stone (r, q)
            min_r, max_r = 0, N
            min_q, max_q = 0, N
            
            offsets = [(0,0), (dr, dq)] + intermediates
            
            # Adjust bounds based on offsets
            for or_, oq in offsets:
                # We need 0 <= r + or_ < N
                min_r = max(min_r, -or_)
                max_r = min(max_r, N - or_)
                min_q = max(min_q, -oq)
                max_q = min(max_q, N - oq)
            
            if min_r >= max_r or min_q >= max_q:
                continue
                
            # Base slice for start stone
            base_slice = (slice(min_r, max_r), slice(min_q, max_q))
            
            # Check start stone
            valid_bridges = player_mask[base_slice]
            
            # Check target stone
            target_slice = (slice(min_r + dr, max_r + dr), slice(min_q + dq, max_q + dq))
            valid_bridges = valid_bridges & player_mask[target_slice]
            
            # Check intermediates
            for ir, iq in intermediates:
                int_slice = (slice(min_r + ir, max_r + ir), slice(min_q + iq, max_q + iq))
                valid_bridges = valid_bridges & empty_mask[int_slice]
            
            # Mark the Plane
            if np.any(valid_bridges):
                for ir, iq in intermediates:
                    dest_slice = (slice(min_r + ir, max_r + ir), slice(min_q + iq, max_q + iq))
                    bridge_plane[dest_slice] = np.maximum(bridge_plane[dest_slice], valid_bridges)

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
    board[4,5] = 1
    board[4,7] = 1
    board[4,8] = -1
    board[4,9] = 1

    board[5,2] = -1
    board[5,5] = -1
    board[5,7] = -1
    board[5,8] = -1
    board[5,9] = 1

    board[6,3] = -1
    board[6,6] = -1
    board[6,7] = 1
    board[6,8] = -1
    board[6,9] = 1

    board[7,1] = 1

    board[8,7] = -1
    board[8,8] = 1