from snake import Snake
from utils import Direction
from board import Board, STARTING_POSITION, FOOD_STARTING_POSITION
import pygame
from pygame.locals import *

SNAKE_SPEED = 1000 / 7 # in squares per millisecond
MOVE_EVENT = pygame.USEREVENT + 1

class Game:

    def __init__(self):
        self.game_on = False
        self.score = 0
        self.snake = Snake(STARTING_POSITION, Direction.EAST)
        self.board = Board(self.snake)
        self.food_position = FOOD_STARTING_POSITION
        pygame.init()

    def snake_is_legal(self) -> bool:
        head = self.snake.positions[0]
        return self.board.containes_position(head) and head not in self.snake.positions[1:]
    
    def place_food(self):
        self.food_position = self.board.get_random_position()
    
    def play(self):
        self.board.show_board(self.food_position)
        while not self.game_on:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_RIGHT:
                    self.game_on = True
                    pygame.time.set_timer(MOVE_EVENT, int(SNAKE_SPEED))
                    if self.snake.move(self.food_position):
                        self.place_food()
                    self.board.show_board(self.food_position)

        while self.game_on:
            for event in pygame.event.get():
                if event.type == MOVE_EVENT:
                    snake_ate = self.snake.move(self.food_position)
                    if snake_ate:
                        self.place_food()
                        self.score += 1

                    if not self.snake_is_legal():
                        self.game_on = False
                        continue
                    self.board.show_board(self.food_position)

                elif event.type == KEYDOWN:
                    new_direction = Direction.direction_from_key(event.key)
                    if new_direction != None:
                        self.snake.updateDirection(new_direction)

                    if event.key == K_BACKSPACE:
                        print('backspace')
                        self.game_over = True
                        
                # Check for QUIT event
                elif event.type == QUIT:
                    self.game_over = True
                    
            

        print("Score: " + str(self.score))



if __name__ == '__main__':
    game = Game()
    game.play()
