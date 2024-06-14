from utils import Position, Direction

class Snake:
    def __init__(self, starting_position: Position, direction: Direction):
        self.direction = direction
        self.last_direction_moved = None
        self.positions = [starting_position, starting_position - Position(1, 0), starting_position - Position(2, 0)]

    def updateDirection(self, new_direction):
        if new_direction != self.last_direction_moved.opposite():
            self.direction = new_direction

    # returns if he ate
    def move(self, food: Position) -> bool:
        next_head = self.positions[0].next(self.direction)
        self.positions.insert(0, next_head)
        is_eating = self.is_eating(food)
        if not is_eating:
            self.positions.pop()
        self.last_direction_moved = self.direction
        return is_eating


    def is_eating(self, food_position: Position):
        return self.positions[0] == food_position



