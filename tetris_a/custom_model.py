import numpy as np
import torch
from agent import DQNAgent

class CUSTOM_AI_MODEL:
    def __init__(self):
        self.agent = None

    def load_agent(self):
        if self.agent is not None:
            return
        
        self.agent = DQNAgent()
        self.agent.load("saved_models/dqn_final.pth")
        self.agent.epsilon = 0.0

    def extract_state(self, board, piece):
        heights = np.array(board.heights, dtype=np.float32)

        holes = []
        for col in range(board.width):
            found_block = False
            count = 0

            for row in range(board.height):
                if board.board[row][col]:
                    found_block = True
                elif found_block:
                    count += 1

            holes.append(count)

        holes = np.array(holes, dtype=np.float32)

        aggregate_height = heights.sum()
        max_height = heights.max()
        bumpiness = np.abs(np.diff(heights)).sum()

        shapes = [
            ((0, 0), (0, 1), (0, 2), (0, 3)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (1, 0), (1, 1), (2, 0)),
            ((0, 0), (0, 1), (0, 2), (1, 0)),
            ((0, 0), (1, 0), (1, 1), (1, 2)),
            ((0, 0), (1, 0), (1, 1), (2, 1)),
            ((0, 1), (1, 0), (1, 1), (2, 0))
        ]

        def normalize(coords):
            min_x = min(x for x, y in coords)
            min_y = min(y for x, y in coords)
            return tuple(sorted(
                (x - min_x, y - min_y) for x, y in coords
            ))

        piece_shape = normalize(piece.body)
        piece_features = np.zeros(7, dtype=np.float32)

        for i, shape in enumerate(shapes):
            if piece_shape == normalize(shape):
                piece_features[i] = 1
                break

        state = np.concatenate([
            heights / 20,
            holes / 20,
            np.array([
                aggregate_height / 200,
                max_height / 20,
                bumpiness / 180
            ], dtype=np.float32),
            piece_features
        ])

        return state.astype(np.float32)

    def get_valid_actions(self, board, original_piece):
        valid_actions = []
        current_piece = original_piece
        seen = set()

        for rotation in range(4):
            signature = tuple(sorted(current_piece.body))

            if signature not in seen:
                seen.add(signature)

                width = len(current_piece.skirt)

                for x in range(board.width - width + 1):
                    y = board.drop_height(current_piece, x)

                    valid = True

                    for bx, by in current_piece.body:
                        tx = x + bx
                        ty = y + by

                        if (
                            tx < 0 or
                            tx >= board.width or
                            ty < 0 or
                            ty >= board.height + 4 or
                            board.board[ty][tx]
                        ):
                            valid = False
                            break

                    if valid:
                        valid_actions.append(rotation * 10 + x)

            current_piece = current_piece.get_next_rotation()
        return valid_actions

    def decode_action(self, action, original_piece):
        rotation = action // 10
        x = action % 10

        piece = original_piece

        for _ in range(rotation):
            piece = piece.get_next_rotation()

        return x, piece

    def get_best_move(self, board, curr_piece):
        self.load_agent()
        state = self.extract_state(board, curr_piece)
        valid_actions = self.get_valid_actions(board, curr_piece)

        if not valid_actions:
            return 0, curr_piece

        action = self.agent.select_action(state, valid_actions)
        x, piece = self.decode_action(action, curr_piece)
        
        return x, piece