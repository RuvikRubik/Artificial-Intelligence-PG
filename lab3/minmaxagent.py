import random
import time

from connect4 import Connect4
from exceptions import AgentException


class MinMaxAgent:
    def __init__(self, my_token='o', depth=4):
        self.my_token = my_token
        self.depth = depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.minimax(connect4, 1, 0)[1]

    def minimax(self, connect4, x, depth):
        if self.depth == depth or connect4.game_over:
            return self.evaluate(connect4), None

        best_value = -1 if x == 1 else 1
        best_move = None
        for possible_move in connect4.possible_drops():
            new_connect4 = self.simulate_move(connect4, possible_move)
            value, _ = self.minimax(new_connect4, -x, depth + 1)
            if x == 1:
                best_value = max(value, best_value)
            else:
                best_value = min(value, best_value)
            if value == best_value:
                best_move = possible_move
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