from enum import Enum
from pygame.locals import *

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
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
        return Direction((self.value + 2) % 4)
        
    def left_turn(self):
        return Direction((self.value - 1) % 4)
        
    def right_turn(self):
        return Direction((self.value + 1) % 4)

class State():
    num_possible = 512
    def __init__(
            self, 
            direction: Direction = None, 
            danger_straight: bool = None, 
            danger_right: bool = None, 
            danger_left: bool = None, 
            food_north: bool = None, 
            food_south: bool = None, 
            food_east: bool = None, 
            food_west: bool = None,
            state_int: int = None
        ):
        if state_int == None:
            self.direction = direction
            self.danger_straight = danger_straight
            self.danger_right = danger_right
            self.danger_left = danger_left
            self.food_north = food_north
            self.food_south = food_south
            self.food_east = food_east
            self.food_west = food_west
        else:
            binary_state = bin(state_int)
            self.direction = Direction(int(binary_state[2:4], base=2))
            self.danger_straight = binary_state[4:5] == '1'
            self.danger_right = binary_state[5:6] == '1'
            self.danger_left = binary_state[6:7] == '1'
            self.food_north = binary_state[7:8] == '1'
            self.food_south = binary_state[8:9] == '1'
            self.food_east = binary_state[9:10] == '1'
            self.food_west = binary_state[10:11] == '1'

    def get_int(self):
        return int(
                str(bin(self.direction.value)) + 
                str(int(self.danger_straight)) +
                str(int(self.danger_right)) +
                str(int(self.danger_left)) +
                str(int(self.food_north)) +
                str(int(self.food_south)) +
                str(int(self.food_east)) +
                str(int(self.food_west)),
                base = 2
            )
    
    def __str__(self):
        return "\ndirection: " + str(self.direction) +\
                "\ndanger straight: " + str(int(self.danger_straight)) +\
                "\ndanger right: " + str(int(self.danger_right)) +\
                "\ndanger left: " + str(int(self.danger_left)) +\
                "\nfood north: " + str(int(self.food_north)) +\
                "\nfood south: " + str(int(self.food_south)) +\
                "\nfood east: " + str(int(self.food_east)) +\
                "\nfood west: " + str(int(self.food_west)) + "\n"


class Actions(Enum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2

    def apply_to_direction(self, direction: Direction):
        if self == Actions.STRAIGHT:
            return direction
        elif self == Actions.RIGHT:
            return direction.right_turn()
        elif self == Actions.LEFT:
            return direction.left_turn()
        

class GameColor(Enum):
    SNAKE = (0, 255, 0)
    FOOD = (255, 0, 0)
    BACKGROUND = (0, 0, 0)
    PATH = (50, 50, 50)

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
        
    def direction_of(self, position) -> Direction:
        if self.x == position.x:
            if self.y > position.y:
                return Direction.NORTH
            elif self.y < position.y:
                return Direction.SOUTH
        elif self.y == position.y:
            if self.x < position.x:
                return Direction.EAST
            elif self.x > position.x:
                return Direction.WEST
        else:
            print("Inavlid input to direction of")
            print(str(self) + ' and ' + str(position))
    
    def __add__(self, o):
        return Position(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Position(self.x - o.x, self.y - o.y)
    
    def __eq__(self, o):
        return self.x == o.x and self.y == o.y
    
    def __hash__(self):
        return int(0.5 * ((self.x + self.y) * (self.x + self.y + 1)) + self.y)
    
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'