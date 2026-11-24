import numpy as np
from piece import Piece
from game import Game

class TetrisEnvironment:
    def __init__(self):
        self.game = None
        self.reset()

    def reset(self):
        self.game = Game("student")
        return self.extract_state()

    def extract_state(self):
        board = self.game.board
        piece = self.game.curr_piece

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
        holes = np.array(holes)

        aggregate_height = heights.sum()
        max_height = heights.max()

        bumpiness = np.abs(
            np.diff(heights)
        ).sum()

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
                (x - min_x, y - min_y)
                for x, y in coords
            ))

        piece_shape = normalize(piece.body)
        piece_features = np.zeros(7)

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
            ]),

            piece_features

        ])

        return state.astype(np.float32)

    def get_valid_actions(self):
        board = self.game.board
        original_piece = self.game.curr_piece
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
                        action = rotation * 10 + x
                        valid_actions.append(action)

            current_piece = (current_piece.get_next_rotation())

        return valid_actions

    def decode_action(self, action):
        rotation = action // 10
        x = action % 10
        piece = self.game.curr_piece

        for _ in range(rotation):
            piece = piece.get_next_rotation()
        
        return x, piece

    def step(self, action):
        old_rows = self.game.rows_cleared
        old_heights = np.array(self.game.board.heights, dtype=np.float32)
        old_holes = self.extract_state()[10:20].sum()
        old_aggregate_height = old_heights.sum()
        old_bumpiness = np.abs(np.diff(old_heights)).sum()
    
        x, piece = self.decode_action(action)
        y = self.game.board.drop_height(piece, x)
    
        self.game.curr_piece = piece
        self.game.drop(y, x=x)
    
        done = self.game.board.top_filled()
        lines_cleared = (self.game.rows_cleared - old_rows)
    
        new_heights = np.array(self.game.board.heights, dtype=np.float32)
        new_holes = (0 if done else self.extract_state()[10:20].sum())
        new_aggregate_height = new_heights.sum()
        new_bumpiness = np.abs(np.diff(new_heights)).sum()
        
        line_reward = 10.0 * lines_cleared
        height_penalty = (0.05 * (new_aggregate_height - old_aggregate_height))
        hole_penalty = (2.0 * (new_holes - old_holes))
        bumpiness_penalty = (0.2 * (new_bumpiness - old_bumpiness))
    
        reward = (
            line_reward
            - height_penalty
            - hole_penalty
            - bumpiness_penalty
        )

        if done:
            reward -= 20.0
    
        next_state = (np.zeros(30, dtype=np.float32) if done else self.extract_state())
    
        return (
            next_state,
            float(reward),
            done,
            {
                "lines_cleared": lines_cleared,
                "pieces_dropped": self.game.pieces_dropped,
                "aggregate_height": float(new_aggregate_height),
                "holes": float(new_holes),
                "bumpiness": float(new_bumpiness),
                "line_reward": float(line_reward),
                "height_penalty": float(height_penalty),
                "hole_penalty": float(hole_penalty),
                "bumpiness_penalty": float(bumpiness_penalty)
            }
        )