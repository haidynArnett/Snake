from enum import Enum
from pygame.locals import *

class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

    def direction_from_key(key):
        if key == K_UP:
            return Direction.NORTH
        elif key == K_DOWN:
            return Direction.SOUTH
        elif key == K_RIGHT:
            return Direction.EAST
        elif key == K_LEFT:
            return Direction.WEST
        else:
            return None

    def opposite(self):
        if self == Direction.NORTH:
            return Direction.SOUTH
        elif self == Direction.SOUTH:
            return Direction.NORTH
        elif self == Direction.EAST:
            return Direction.WEST
        elif self == Direction.WEST:
            return Direction.EAST
            

class GameColor(Enum):
    SNAKE = (0, 255, 0)
    FOOD = (255, 0, 0)
    BACKGROUND = (0, 0, 0)

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def next(self, direction: Direction):
        if direction == Direction.NORTH:
            return Position(self.x, self.y - 1)
        elif direction == Direction.SOUTH:
            return Position(self.x, self.y + 1)
        elif direction == Direction.EAST:
            return Position(self.x + 1, self.y)
        elif direction == Direction.WEST:
            return Position(self.x - 1, self.y)
    
    def __add__(self, o):
        return Position(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Position(self.x - o.x, self.y - o.y)
    
    def __eq__(self, o):
        return self.x == o.x and self.y == o.y
    
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'