from utils import Position, Direction, State, Actions
from snake import Snake
from board import Board
import random

class Player:
    def __init__(self, snake: Snake):
        self.snake = snake
        self.path = []

    def update_game_status(self, food_position: Position):
        pass

    def process_direction_input(self, direction: Direction):
        pass

    def get_next_direction(self) -> Direction:
        pass


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

    def update_game_status(self, food_position: Position):
        frontier = [([], self.snake.positions[0])]
        visited_set = set()
        visited_set.add(self.snake.positions[0])
        while len(frontier) != 0:
            path, position = frontier.pop(0)
            if position == food_position:
                self.path = path + [food_position]
                return
            for next_position in self.board.get_possible_next_positions(position):
                if next_position not in visited_set:
                    visited_set.add(next_position)
                    frontier.append((path + [next_position], next_position))
        print("Path does not exist")
        return []
    
    def get_next_direction(self) -> Direction:
        if len(self.path) == 0:
            return None 
        next_direction = self.snake.positions[0].direction_of(self.path.pop(0))
        return next_direction
            
class QLearningAgent(Player):
    def __init__(self, snake: Snake, board: Board):
        super().__init__(snake)
        self.board = board
        self.next_direction = None
        self.q_table = [[0 for _ in range(len(Actions))] for _ in range(State.num_possible)]
        self.learning_rate = 0.55
        self.discount_factor = 0.5
        self.exploration_prob = 0.2
        self.epochs = 1000
        self.exploit = False

    def update_game_status(self, food_position: Position):
        self.food_position = food_position
        current_state = self.get_state()

        if random.random() > self.exploration_prob or self.exploit:
            action = Actions(max(range(len(Actions)), key=lambda a : self.q_table[current_state.get_int()][a]))
        else:
            action = Actions(random.randint(0, len(Actions) - 1))
        self.next_direction = action.apply_to_direction(self.snake.direction)

        # determine reward
        next_head_position = self.snake.positions[0].next(self.next_direction)
        reward = -0.001
        # reward = 0
        if next_head_position in self.snake.positions[0:-1] or not self.board.contains_position(next_head_position):
            reward = -1
        elif next_head_position == self.food_position:
            reward = 1

        # update q table
        max_adjacent_q_value = max([self.q_table[self.get_next_state().get_int()][action_int] for action_int in range(len(Actions))])
        self.q_table[current_state.get_int()][action.value] += self.learning_rate * (reward + self.discount_factor * max_adjacent_q_value - self.q_table[current_state.get_int()][action.value])
    
    def get_next_direction(self):
        return self.next_direction
    
    def get_state(self):
        straight_position = self.snake.positions[0].next(self.snake.direction)
        right_position = self.snake.positions[0].next(self.snake.direction.right_turn())
        left_position = self.snake.positions[0].next(self.snake.direction.left_turn())
        return State(
            self.snake.direction,
            danger_straight = not self.board.contains_position(straight_position) or 
                straight_position in self.snake.positions[0:-1],
            danger_right = not self.board.contains_position(right_position) or
                right_position in self.snake.positions[0:-1],
            danger_left = not self.board.contains_position(left_position) or
                left_position in self.snake.positions[0:-1],
            food_north = self.food_position.y < self.snake.positions[0].y,
            food_south = self.food_position.y > self.snake.positions[0].y,
            food_east = self.food_position.x > self.snake.positions[0].x,
            food_west = self.food_position.x < self.snake.positions[0].x
        )
    
    def get_next_state(self):
        next_head_position = self.snake.positions[0].next(self.next_direction)
        straight_position = next_head_position.next(self.next_direction)
        right_position = next_head_position.next(self.next_direction.right_turn())
        left_position = next_head_position.next(self.next_direction.left_turn())
        return State(
            self.next_direction,
            danger_straight = not self.board.contains_position(straight_position) or 
                straight_position in self.snake.positions[0:-2],
            danger_right = not self.board.contains_position(right_position) or
                right_position in self.snake.positions[0:-2],
            danger_left = not self.board.contains_position(left_position) or
                left_position in self.snake.positions[0:-2],
            food_north = self.food_position.y < next_head_position.y,
            food_south = self.food_position.y > next_head_position.y,
            food_east = self.food_position.x > next_head_position.x,
            food_west = self.food_position.x < next_head_position.x
        )


