import random
from connect4 import Connect4
from exceptions import AgentException


class AlphaBetaAgent:
    def __init__(self, my_token='o', depth=4):
        self.my_token = my_token
        self.depth = depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.alphabeta(connect4, self.depth, float('-inf'), float('inf'))[1]

    def alphabeta(self, connect4, depth, alpha, beta):
        if connect4.game_over:
            return self.evaluate(connect4), None
        if depth == 0:
            my_score = 0
            center_column = connect4.center_column()
            my_score += center_column.count(self.my_token)
            return my_score, None

        best_value = float('-inf') if connect4.who_moves == self.my_token else float('inf')
        best_move = None

        for possible_move in connect4.possible_drops():
            new_connect4 = self.simulate_move(connect4, possible_move)
            value, _ = self.alphabeta(new_connect4, depth - 1, alpha, beta)

            if connect4.who_moves == self.my_token:
                if value > best_value:
                    best_value = value
                    best_move = possible_move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                if value < best_value:
                    best_value = value
                    best_move = possible_move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return best_value, best_move

    def evaluate(self, connect4):
        if connect4.wins == self.my_token:
            return 1
        elif connect4.wins != self.my_token and connect4.wins is not None:
            return -1
        else:
            return 0

    def simulate_move(self, connect4, move):
        new_connect4 = Connect4(width=connect4.width, height=connect4.height)
        new_connect4.board = [row[:] for row in connect4.board]
        new_connect4.who_moves = connect4.who_moves
        new_connect4.game_over = connect4.game_over
        new_connect4.wins = connect4.wins
        new_connect4.drop_token(move)
        return new_connect4
