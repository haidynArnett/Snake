from utils import Position, Direction
from snake import Snake
from board import Board
import time


class Player:
    def __init__(self, snake: Snake):
        self.snake = snake
        self.path = []

    def update_path(self, food_position: Position):
        self.path = self.find_path(food_position)

    def find_path(self, food_position: Position) -> list:
        pass

    def process_direction_input(self, direction: Direction):
        pass

    def get_next_direction(self) -> Direction:
        if len(self.path) == 0:
            return None 
        return self.snake.positions[0].direction_of(self.path[0])


class Human(Player):
    def __init__(self, snake: Snake):
        super().__init__(snake)
        self.direction = Direction.EAST
        
    def process_direction_input(self, direction: Direction):
        self.direction = direction

    def get_next_direction(self) -> Direction:
        return self.direction
    

class BFSAgent(Player):
    def __init__(self, snake: Snake, board: Board):
        super().__init__(snake)
        self.board = board

    def find_path(self, food_position: Position) -> list:
        frontier = [([], self.snake.positions[0])]
        visited_set = set()
        visited_set.add(self.snake.positions[0])
        while len(frontier) != 0:
            path, position = frontier.pop(0)
            if position == food_position:
                return path + [food_position]
            for next_position in self.board.get_next_positions(position):
                if next_position not in visited_set:
                    visited_set.add(next_position)
                    frontier.append((path + [next_position], next_position))
        print("Path does not exist")
        return []
            


