from snake import Snake
from utils import Position, Direction, GameColor
import random
import pygame
from pygame.locals import *

SQUARE_SIZE = 20
BOARD_WIDTH = 10
BOARD_HEIGHT = 10
STARTING_POSITION = Position(5,5)
FOOD_STARTING_POSITION = Position(8, 5)

class Board:
    def __init__(self,snake: Snake):
        self.snake = snake
        self.screen = pygame.display.set_mode((SQUARE_SIZE * BOARD_WIDTH, SQUARE_SIZE * BOARD_HEIGHT))
        self.screen.fill(GameColor.BACKGROUND.value)

    def get_random_position(self):
        position = None
        while position is None or position in self.snake.positions:
            position = Position(random.randint(0, BOARD_WIDTH - 1), random.randint(0, BOARD_HEIGHT - 1))
        return position
    
    def get_possible_next_positions(self, position: Position) -> Position:
        next_positions = []
        for direction in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            next = position.next(direction)
            if self.contains_position(next) and next not in self.snake.positions:
                next_positions.append(next)
        return next_positions

    def contains_position(self, p: Position) -> bool:
        return p.x >= 0 and p.x < BOARD_WIDTH and p.y >= 0 and p.y < BOARD_HEIGHT

    def show_board(self, food_position, path):
        self.screen.fill(GameColor.BACKGROUND.value)
        if path is not None:
            for position in path:
                self.draw_square(position, GameColor.PATH)
        self.draw_square(food_position, GameColor.FOOD)
        for position in self.snake.positions:
            self.draw_square(position, GameColor.SNAKE)
        pygame.display.flip()

    def draw_square(self, position: Position, color: GameColor):
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
        surface.fill(color.value)
        self.screen.blit(surface, (position.x * SQUARE_SIZE, position.y * SQUARE_SIZE))