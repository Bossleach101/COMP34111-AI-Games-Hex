import math

class Node:
    def __init__(self, state, parent=None, move=None, prior=1.0, use_move_heuristic=True):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = prior  # Prior probability for PUCT
        self.use_move_heuristic = use_move_heuristic
        self.untried_moves = state.get_valid_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0 and len(self.children) > 0

    def is_terminal(self):
        return self.state.is_terminal()

    def best_child(self, c_param=1.41):
        """Select best child using PUCT formula (Predictor + UCT)"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                q_value = 0
            else:
                q_value = child.value / child.visits

            # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            u_value = c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            puct_value = q_value + u_value
            choices_weights.append(puct_value)

        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """Expand node by creating ONE child (traditional MCTS)"""
        if self.is_fully_expanded() or self.is_terminal():
            return None

        if not self.untried_moves:
            return None

        # Pick one untried move
        move = self.untried_moves.pop(0)

        # Create new child state
        new_state = self.state.copy()
        player = new_state.get_current_player()
        new_state.make_move(move, player)

        # Calculate prior (uniform for now)
        total_moves = len(self.children) + len(self.untried_moves) + 1
        uniform_prior = 1.0 / total_moves if total_moves > 0 else 1.0

        # Create and add child
        child = Node(new_state, parent=self, move=move, prior=uniform_prior,
                    use_move_heuristic=self.use_move_heuristic)
        self.children.append(child)

        return child

    def expand_with_policy(self, policy_vector):
        """
        Expand node by creating all children, using the policy vector for priors.
        policy_vector: 11x11 numpy array of probabilities
        """
        if self.is_fully_expanded() or self.is_terminal():
            return

        valid_moves = self.untried_moves.copy()
        
        # Create all children
        for move in valid_moves:
            r, c = move
            prior = policy_vector[r, c]
            
            new_state = self.state.copy()
            player = new_state.get_current_player()
            new_state.make_move(move, player)

            child = Node(new_state, parent=self, move=move, prior=prior,
                        use_move_heuristic=self.use_move_heuristic)
            self.children.append(child)

        # Clear untried moves since we've expanded all
        self.untried_moves = []