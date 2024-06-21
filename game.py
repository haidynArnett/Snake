from snake import Snake
from utils import Direction
from board import Board, STARTING_POSITION, FOOD_STARTING_POSITION
from agents import Player, Human, BFSAgent
import pygame
from pygame.locals import *

# SNAKE_SPEED = 1000 / 7 # in squares per millisecond
SNAKE_SPEED = 1000 / 20
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
        return self.board.containes_position(head) and head not in self.snake.positions[1:]
    
    def place_food(self):
        self.food_position = self.board.get_random_position()
    
    def play(self):
        self.player.update_path(self.food_position)
        self.board.show_board(self.food_position, self.player.path)
        while not self.game_on:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_RIGHT:
                    self.game_on = True
                    pygame.time.set_timer(MOVE_EVENT, int(SNAKE_SPEED))
                    self.snake.move(self.food_position)
                    self.player.update_path(self.food_position)
                    self.board.show_board(self.food_position, self.player.path)

        while self.game_on:
            for event in pygame.event.get():
                if event.type == MOVE_EVENT:
                    self.snake.updateDirection(self.player.get_next_direction())
                    snake_ate = self.snake.move(self.food_position)
                    if snake_ate:
                        self.place_food()
                        self.player.update_path(self.food_position)
                        self.score += 1

                    if not self.snake_is_legal():
                        self.game_on = False
                        continue
                    self.player.update_path(self.food_position)
                    self.board.show_board(self.food_position, self.player.path)

                elif event.type == KEYDOWN:
                    new_direction = Direction.direction_from_key(event.key)
                    if new_direction != None:
                        player.process_direction_input(new_direction)

                    if event.key == K_BACKSPACE:
                        print('backspace')
                        self.game_over = True
                        
                # Check for QUIT event
                elif event.type == QUIT:
                    self.game_over = True

        print("Score: " + str(self.score))



if __name__ == '__main__':
    snake = Snake(STARTING_POSITION, Direction.EAST)
    board = Board(snake)
    # player = Human(snake)
    player = BFSAgent(snake, board)
    game = Game(snake, player, board)
    game.play()
