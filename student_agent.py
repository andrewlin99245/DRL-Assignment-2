import math
import struct
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import gc
import time
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved
        if moved:
           self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

# --- Grid Transformation Utilities ---

def flip_horizontally(flat_list):
    # Reshape flat_list into a 4×4 matrix, reverse each row, and flatten the result.
    matrix = [flat_list[i * 4: (i + 1) * 4] for i in range(4)]
    flipped = [row[::-1] for row in matrix]
    return [elem for row in flipped for elem in row]

def rotate_clockwise(flat_list):
    # Rotates a 4×4 matrix (represented as a flat list) 90° clockwise.
    matrix = [flat_list[i * 4: (i + 1) * 4] for i in range(4)]
    rotated = [[matrix[3 - j][i] for j in range(4)] for i in range(4)]
    return [value for row in rotated for value in row]

def perform_rotation(flat_list, times):
    # Execute the clockwise rotation multiple times.
    result = flat_list.copy()
    for _ in range(times):
        result = rotate_clockwise(result)
    return result

# --- Feature Extraction Module ---

class SymmetryFeature:
    def __init__(self, positions, num_symmetries=8):
        self.positions = positions
        self.num_symmetries = num_symmetries
        self.num_entries = 1 << (len(positions) * 4)
        self.parameters = [0.0] * self.num_entries
        self.symmetry_variants = self._compute_variants()

    def _compute_variants(self):
        original_order = list(range(16))
        variants = []
        for s in range(self.num_symmetries):
            mapping = original_order.copy()
            if s >= 4:
                mapping = flip_horizontally(mapping)
            mapping = perform_rotation(mapping, s % 4)
            variant = [mapping[i] for i in self.positions]
            variants.append(variant)
        return variants

    def compute_indices(self, board_flat):
        indices = []
        for variant in self.symmetry_variants:
            idx = 0
            for pos, pos_index in enumerate(variant):
                idx |= board_flat[pos_index] << (4 * pos)
            indices.append(idx)
        return indices

    def predict(self, board_flat):
        idx_list = self.compute_indices(board_flat)
        return sum(self.parameters[i] for i in idx_list)

    def adjust(self, board_flat, delta):
        idx_list = self.compute_indices(board_flat)
        modification = delta / len(idx_list)
        for i in idx_list:
            self.parameters[i] += modification
        return sum(self.parameters[i] for i in idx_list)

    def label(self):
        hex_label = "".join(f"{pos:x}" for pos in self.positions)
        return f"{len(self.positions)}-tuple {hex_label}"

    def load_from_stream(self, stream, count_format='I'):
        # Read and discard the stored name length and name.
        name_len_data = stream.read(4)
        name_len = struct.unpack('i', name_len_data)[0]
        stream.read(name_len)
        # Read the number of parameters.
        format_size = struct.calcsize(count_format)
        param_count_data = stream.read(format_size)
        param_count = struct.unpack(count_format, param_count_data)[0]
        float_size = struct.calcsize('f')
        param_data = stream.read(param_count * float_size)
        self.parameters = list(struct.unpack(f'{param_count}f', param_data))

# --- NTuple System for Approximating Values ---

class NTupleSystem:
    def __init__(self, board_dimension, pattern_list, num_symmetries=8):
        self.board_dim = board_dimension
        self.features = [SymmetryFeature(p, num_symmetries) for p in pattern_list]

    def estimate_board(self, board_flat):
        return sum(feature.predict(board_flat) for feature in self.features)

    def update_parameters(self, board_flat, delta, learning_rate):
        for feature in self.features:
            feature.adjust(board_flat, learning_rate * delta)

    def load_parameters(self, filepath, count_format='Q'):
        format_size = struct.calcsize(count_format)
        with open(filepath, 'rb') as f:
            _ = struct.unpack(count_format, f.read(format_size))[0]  # Skip header count
            for feature in self.features:
                feature.load_from_stream(f, count_format)

# --- Board Processing Helpers ---

def tile_to_exponent(tile):
    return 0 if tile == 0 else int(math.log(tile, 2))

def flatten_grid(board_array):
    flat_list = []
    for row in range(board_array.shape[0]):
        for col in range(board_array.shape[1]):
            flat_list.append(tile_to_exponent(board_array[row, col]))
    return flat_list

def condense_line(line):
    non_zero = line[line != 0]
    return np.pad(non_zero, (0, 4 - len(non_zero)), mode='constant')

def merge_line(line):
    accumulated = 0
    new_line = line.copy()
    for j in range(3):
        if new_line[j] != 0 and new_line[j] == new_line[j + 1]:
            new_line[j] *= 2
            accumulated += new_line[j]
            new_line[j + 1] = 0
    return new_line, accumulated

def collapse_and_merge(line):
    compressed = condense_line(line)
    merged, points = merge_line(compressed)
    final_line = condense_line(merged)
    return final_line, points

def move_left(board):
    board_copy = board.copy()
    score_total = 0
    for i in range(4):
        current_row = board_copy[i, :].copy()
        new_row, points = collapse_and_merge(current_row)
        score_total += points
        board_copy[i, :] = new_row
    return board_copy, score_total

def move_right(board):
    board_copy = board.copy()
    score_total = 0
    for i in range(4):
        current_row = board_copy[i, ::-1].copy()
        new_row, points = collapse_and_merge(current_row)
        score_total += points
        board_copy[i, :] = new_row[::-1]
    return board_copy, score_total

def move_up(board):
    board_copy = board.copy()
    score_total = 0
    for j in range(4):
        column = board_copy[:, j].copy()
        column = condense_line(column)
        merged, points = merge_line(column)
        final_col = condense_line(merged)
        score_total += points
        board_copy[:, j] = final_col
    return board_copy, score_total

def move_down(board):
    board_copy = board.copy()
    score_total = 0
    for j in range(4):
        column = board_copy[::-1, j].copy()
        column = condense_line(column)
        merged, points = merge_line(column)
        final_col = condense_line(merged)
        score_total += points
        board_copy[:, j] = final_col[::-1]
    return board_copy, score_total

def perform_move(board, direction):
    if direction == 0:
        return move_up(board)
    elif direction == 1:
        return move_down(board)
    elif direction == 2:
        return move_left(board)
    elif direction == 3:
        return move_right(board)

def can_move(board, direction):
    new_board, _ = perform_move(board, direction)
    return not np.array_equal(board, new_board)

def add_tile(board):
    empties = list(zip(*np.where(board == 0)))
    if not empties:
        return board.copy()
    new_board = board.copy()
    i, j = random.choice(empties)
    new_board[i, j] = 2 if random.random() < 0.9 else 4
    return new_board

def choose_best(board, system):
    optimal_direction = None
    optimal_value = -float('inf')
    for direction in range(4):
        next_board, _ = perform_move(board, direction)
        if np.array_equal(board, next_board):
            continue
        flat_next = flatten_grid(next_board)
        value = system.estimate_board(flat_next)
        if value > optimal_value:
            optimal_value = value
            optimal_direction = direction
    return optimal_direction, optimal_value

# --- MCTS Tree Structures ---

class StateNode:
    def __init__(self, board, parent=None):
        self.kind = 0  # 0 for a regular state node
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # mapping: action -> AfterMoveNode
        self.visits = 0
        self.total = 0.0

    def is_expanded(self, moves):
        return all(move in self.children for move in moves)

    def select_action(self, moves, exploration=1.0):
        best_move = None
        best_score = -float('inf')
        for move in moves:
            if move not in self.children:
                return move
            child = self.children[move]
            score = child.total + exploration * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

class AfterMoveNode:
    def __init__(self, board, parent=None, reward=0):
        self.kind = 1  # 1 for after-move state
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # mapping: (i, j, tile) -> StateNode
        self.visits = 0
        self.total = 0.0
        self.immediate = reward

    def is_expanded(self, free_slots):
        # Two possibilities (tile 2 or 4) per free slot.
        return len(self.children) == len(free_slots) * 2

# --- Temporal-Difference MCTS Algorithm ---

class TemporalMCTS:
    def __init__(self, ntuple_system, iterations=1000, exploration_factor=1.0, scaling=4096):
        self.ntuple_system = ntuple_system
        self.iterations = iterations
        self.exploration_factor = exploration_factor
        self.scale = scaling

    def traverse(self, root, available_moves=None):
        trajectory = []
        current = root
        moves = available_moves

        while True:
            # State‐node logic
            if current.kind == 0:# before the random tile is added
                if moves is None:
                    tmp_env = self.initialize_env(current.board)
                    moves = [a for a in range(4) if tmp_env.is_move_legal(a)]

                if not current.is_expanded(moves):
                    # would have returned (current, trajectory) here
                    break

                action = current.select_action(moves, self.exploration_factor)
                trajectory.append((current, action))
                current = current.children[action]
                moves = None
                continue

        # Afterstate‐node logic
            if current.kind == 1:
                free_cells = list(zip(*np.where(current.board == 0)))
                if not free_cells or current.is_expanded(free_cells):
                    # would have returned (current, trajectory) here
                    break
                # if there *are* free cells and not fully expanded,
                i, j = random.choice(free_cells)            # pick a random empty cell
                tile = 2 if random.random() < 0.9 else 4     # 90% for 2, 10% for 4
                # descend into that child
                current = current.children[(i, j, tile)]
                continue
            break

        return current, trajectory

    def expand(self, node, env, last_move=None):
        if node.kind == 0:
            moves_possible = [m for m in range(4) if can_move(node.board, m)]
            for move in moves_possible:
                if move not in node.children:
                    new_board, reward = perform_move(node.board, move)
                    new_after_node = AfterMoveNode(new_board, node, reward)
                    node.children[move] = new_after_node
                    return new_after_node
        if node.kind == 1:
            free_cells = list(zip(*np.where(node.board == 0)))
            for (i, j) in free_cells:
                for tile in (2, 4):
                    key = (i, j, tile)
                    if key not in node.children:
                        # spawn the tile and create the next StateNode
                        next_board = node.board.copy()
                        next_board[i, j] = tile
                        child = StateNode(next_board, node)
                        node.children[key] = child
                        return child
        return node

    def simulate(self, node):
        snapshot = node.board
        base_estimate = self.ntuple_system.estimate_board(flatten_grid(snapshot))
        if node.kind == 1:
            return (node.immediate + base_estimate) / self.scale
        legal_moves = [m for m in range(4) if can_move(snapshot, m)]
        if not legal_moves:
            return 0
        best_val = -float('inf')
        # node of type 0
        for move in legal_moves:
            new_board, reward = perform_move(snapshot, move)
            val = self.ntuple_system.estimate_board(flatten_grid(new_board))
            candidate = (reward + val) / self.scale
            best_val = max(best_val, candidate)
        return best_val

    def backpropagate(self, node, sim_val, path):
        node.visits += 1
        node.total += (sim_val - node.total) / node.visits
        for parent, move in reversed(path):
            child = parent.children[move]
            parent.visits += 1
            parent.total += (child.total - parent.total) / parent.visits

    def select_best(self, root, moves):
        best_move = None
        highest_visits = -1
        for move in moves:
            count = 0 if move not in root.children else root.children[move].visits
            if count > highest_visits:
                highest_visits = count
                best_move = move
        return best_move

    def initialize_env(self, board):
        # Assuming Game2048Env implements gym.Env
        environment = Game2048Env()
        environment.board = board.copy()
        return environment
    

patterns = [
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 4, 5, 6],
        [4, 5, 6, 8, 9, 10]
    ]
ntuple_sys = NTupleSystem(board_dimension=4, pattern_list=patterns)
ntuple_sys.load_parameters("my_best_weight.bin")
mcts = TemporalMCTS(ntuple_sys, iterations=100, exploration_factor=1.0, scaling=3584)
def init_model():
    global approximator
    if approximator is None:
        gc.collect() 
        approximator = NTupleSystem(board_dimension=4, pattern_list=patterns)
        approximator.load_parameters("my_best_weight.bin") 
def get_action(state,score):
    # Assume Game2048Env is defined (as a gym.Env)
    root_state = StateNode(state)
    temp_env = mcts.initialize_env(state)
    legal_moves = [m for m in range(4) if temp_env.is_move_legal(m)]
    if not legal_moves:
        selected_move = [m for m in range(4)]
    else:
        for _ in range(mcts.iterations):
            leaf, path_trace = mcts.traverse(root_state, temp_env, legal_moves)
            if leaf.visits > 0:
                leaf = mcts.expand(leaf, temp_env, path_trace[-1][0] if path_trace else None)
            sim_result = mcts.simulate(leaf)
            mcts.backpropagate(leaf, sim_result, path_trace)
        selected_move = mcts.select_best(root_state, legal_moves)
    return selected_move

    
