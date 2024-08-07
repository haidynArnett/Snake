from snake import Snake
from utils import Direction, State, Actions
from board import Board, STARTING_POSITION, FOOD_STARTING_POSITION
from agents import Player, Human, BFSAgent, QLearningAgent, DeepQLearningAgent
import pygame
from pygame.locals import *

SNAKE_SPEED = 0.0075
MOVE_EVENT = pygame.USEREVENT + 1

class Game:

    def __init__(self, snake: Snake, player: Player, board: Board):
        self.game_on = False
        self.score = 0
        self.snake = snake
        self.player = player
        self.board = board
        self.food_position = FOOD_STARTING_POSITION
        pygame.init()

    def snake_is_legal(self) -> bool:
        head = self.snake.positions[0]
        return self.board.contains_position(head) and head not in self.snake.positions[1:]
    
    def place_food(self):
        random_position = self.board.get_random_position()
        while random_position in self.snake.positions:
            random_position = self.board.get_random_position()
        self.food_position = random_position

    def reset(self):
        self.food_position = FOOD_STARTING_POSITION
        self.snake.reset(STARTING_POSITION, Direction.EAST)
        self.score = 0
    
    def play(self, training_mode = False, max_steps = None):
        if training_mode:
            self.game_on = True
            self.place_food()
    
        # show board and wait for keyboard input to start game
        self.board.show_board(self.food_position, self.player.path)
        while not self.game_on:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_RIGHT:
                    self.game_on = True
                    if not training_mode:
                        pygame.time.set_timer(MOVE_EVENT, int(1.0 / SNAKE_SPEED))

        steps = 0
        while self.game_on and (max_steps == None or steps <= max_steps):
            if training_mode == True:
                pygame.event.post(pygame.event.Event(MOVE_EVENT))
            for event in pygame.event.get():
                if event.type == MOVE_EVENT:
                    steps += 1
                    self.player.update_game_status(self.food_position, self.score)
                    self.snake.updateDirection(self.player.get_next_direction())
                    snake_ate = self.snake.move(self.food_position)
                    if snake_ate:
                        self.place_food()
                        self.score += 1

                    if not self.snake_is_legal():
                        self.game_on = False
                        break
                    self.board.show_board(self.food_position, self.player.path)

                elif event.type == KEYDOWN:
                    new_direction = Direction.direction_from_key(event.key)
                    if new_direction != None:
                        player.process_direction_input(new_direction)
                        
                # Check for QUIT event
                elif event.type == QUIT:
                    self.game_over = True

        if max_steps != None and steps > max_steps:
            print("exceded max steps at score: " + str(self.score))
        return self.score

if __name__ == '__main__':
    snake = Snake(STARTING_POSITION, Direction.EAST)
    board = Board(snake)
    # player = Human(snake)
    # player = BFSAgent(snake, board)
    # player = QLearningAgent(snake, board)
    player = DeepQLearningAgent(snake, board)
    game = Game(snake, player, board)

    # player.tune_hyperparameters(game)
    player.train(game)
    
    player.exploit = True
    score = game.play(training_mode=False)
    print("Score: " + str(score))

