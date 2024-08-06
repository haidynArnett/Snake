from utils import Position, Direction

class Snake:
    def __init__(self, starting_position: Position, direction: Direction):
        self.reset(starting_position, direction)

    def reset(self, position: Position, direction: Direction):
        self.direction = direction
        # self.positions = [position]
        self.positions = [position, position - Position(1, 0), position - Position(2, 0)]
        # self.positions = [position, position - Position(1, 0), position - Position(2, 0), position - Position(3, 0), position - Position(4, 0)]

    def updateDirection(self, new_direction):
        if new_direction != None and new_direction != self.direction.opposite():
            self.direction = new_direction
        else:
            print("Can't update to direction: " + str(new_direction))

    # returns if he ate
    def move(self, food: Position) -> bool:
        next_head = self.positions[0].next(self.direction)
        self.positions.insert(0, next_head)
        is_eating = self.is_eating(food)
        if not is_eating:
            self.positions.pop()
        return is_eating


    def is_eating(self, food_position: Position):
        return self.positions[0] == food_position



