from utils import Position, Direction, State, Actions
from snake import Snake
from board import Board
import random
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Player:
    def __init__(self, snake: Snake):
        self.snake = snake
        self.path = []

    def train(self, game):
        pass

    def update_game_status(self, food_position: Position, score: int):
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

    def update_game_status(self, food_position: Position, score: int):
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
        self.learning_rate = 0.4
        self.discount_factor = 0.2
        self.exploration_prob = 1.0
        # self.exploration_decay_rate = 0.9979 # goes with 5000 epochs
        # self.epochs = 5000
        self.exploration_decay_rate = 0.995 # goes with 1000 epochs
        self.epochs = 1000
        self.exploit = False

    def update_game_status(self, food_position: Position, score: int):
        self.food_position = food_position
        current_state = self.get_state()

        if random.random() > self.exploration_prob or self.exploit:
            action = Actions(max(range(len(Actions)), key=lambda a : self.q_table[int(current_state)][a]))
        else:
            action = Actions(random.randint(0, len(Actions) - 1))
        self.next_direction = action.apply_to_direction(self.snake.direction)

        # determine reward
        next_head_position = self.snake.positions[0].next(self.next_direction)
        reward = score
        # reward = 0
        if next_head_position in self.snake.positions[0:-1] or not self.board.contains_position(next_head_position):
            reward = -100_000
            # reward = -100000 * score
            # reward = -2 * score
            # -10
        elif next_head_position == self.food_position:
            reward = 100_000
            # reward = 100000 * score
            # reward = 2 * score
            # reward = 1

        # update q table
        max_adjacent_q_value = max([self.q_table[int(self.get_next_state())][action_int] for action_int in range(len(Actions))])
        self.q_table[int(current_state)][action.value] += self.learning_rate * (reward + self.discount_factor * max_adjacent_q_value - self.q_table[int(current_state)][action.value])

    
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
    
    def print_q_table(self):
        print("q table:")
        for state_int in range(State.num_possible):
            for action_int in range(len(Actions)):
                print("State: " + str(State(state_int=state_int)) + "Action: " + str(Actions(action_int).name) + "\nq value: " + str(player.q_table[state_int][action_int]))
        
    def train(self, game):
        training_scores = []
        for epoch in range(self.epochs):
            score = game.play(training_mode=True, max_steps=10_000)
            print("\nepoch: " + str(epoch))
            print("Score: " + str(score))
            print("exploration probability: " + str(self.exploration_prob))
            training_scores.append(score)
            game.reset()
            self.exploration_prob *= self.exploration_decay_rate

        print("epochs: " + str(self.epochs))
        print("learning rate: " + str(self.learning_rate))
        print("discount factor: " + str(self.discount_factor))
        print("exploration prob: " + str(self.exploration_prob))
        print("training scores: " + "\naverage: " + str(sum(training_scores) / len(training_scores)))
        print("max: " + str(max(training_scores)))
        print("Last 100 average: " + str(sum(training_scores[-100:]) / 100.0))
        print("\n\n")
        self.exploit = True
        return (sum(training_scores[-100:]) / 100.0, (self.epochs, self.learning_rate, self.discount_factor, self.exploration_prob))
    
    def tune_hyperparameters(self):
        # epochss = [1000, 5000, 10_000, 15_000, 20_000]
        # learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # exploration_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # epochss = [1000, 5000, 10_000]
        # learning_rates = [0.2, 0.4, 0.6, 0.8, 1]
        # discount_factors = [0.2, 0.4, 0.6, 0.8, 1]
        # exploration_probs = [0.9]
        # exploration_probs = [0.005, 0.01, 0.05, 0.1]

        epochss = [10_000]
        learning_rates = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
        discount_factors = [0.95]
        exploration_probs = [1]

        results = []
        for epochs in epochss:
            for learning_rate in learning_rates:
                for discount_factor in discount_factors:
                    for exploration_prob in exploration_probs:
                        self.epochs = epochs
                        self.learning_rate = learning_rate
                        self.discount_factor = discount_factor
                        self.exploration_prob = exploration_prob
                        results.append(self.train())
        print("results: " + str(results))
        print("Max: " + str(max(results)))


# class DQN(tf.keras.Model):
#     def __init__(self, num_actions):
#         super(DQN, self).__init__()
#         # self.input_layer = keras.layers.InputLayer()
#         self.dense1 = keras.layers.Dense(24, activation=keras.activations.linear)
#         self.dense2 = keras.layers.Dense(24, activation=keras.activations.sigmoid)
#         # self.dense3 = keras.layers.Dense(128, activation='relu')
#         # self.dense4 = keras.layers.Dense(82, activation='relu')
#         # self.dense5 = keras.layers.Dense(24, activation='relu')
#         self.output_layer = keras.layers.Dense(num_actions, activation=keras.activations.linear)
    
#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         # x = self.dense3(x)
#         # x = self.dense4(x)
#         # x = self.dense5(x)
#         return self.output_layer(x)

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.linear1 = nn.Linear(8, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, num_actions)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DeepQLearningAgent(QLearningAgent):
    def __init__(self, snake: Snake, board: Board):
        super().__init__(snake, board)
        self.q_table = None
        self.learning_rate = 0.001
        # self.learning_rate = 0.0003
        # self.learning_rate = 0.0001
        # self.discount_factor = 0.99
        self.discount_factor = 0.9
        self.exploration_prob = 1
        self.exploration_decay_rate = 0.9985 # goes with 5000 epochs
        self.epochs = 5000
        self.dqn: DQN = DQN(len(Actions))
        # self.loss_function = keras.losses.MeanSquaredError()
        # self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr = self.learning_rate)
        self.criterion = nn.MSELoss()
        self.losses = []

    def update_game_status(self, food_position: Position, score: int):
        self.food_position = food_position
        current_state = self.get_state()

        if random.random() > self.exploration_prob or self.exploit:
            q_values = self.dqn(current_state.tensor_version())
            action = Actions(max(range(len(Actions)), key=lambda a : q_values[a]))
        else:
            action = Actions(random.randint(0, len(Actions) - 1))
        self.next_direction = action.apply_to_direction(self.snake.direction)

        # determine reward
        next_head_position = self.snake.positions[0].next(self.next_direction)
        
        # distance_to_food = abs(self.snake.positions[0].x - self.food_position.x) + abs(self.snake.positions[0].y - self.food_position.y)
        reward = 0
        done = False
        if next_head_position in self.snake.positions[0:-1] or not self.board.contains_position(next_head_position):
            reward = -1
            done = True
        elif next_head_position == self.food_position:
            reward = 1
        
        q_values = self.dqn(current_state.tensor_version())
        target = q_values.clone()
        next_q_values = self.dqn(self.get_next_state().tensor_version())
        new_q = reward + self.discount_factor * max(next_q_values) * (1 - done)
        target[action.value] = new_q

        self.optimizer.zero_grad()
        loss = self.criterion(target, q_values)
        self.losses.append(loss.int())
        loss.backward()
        self.optimizer.step()

        

    def print_q_table(self):
        print("q table:")
        for state_int in range(State.num_possible):
            print("State: " + str(State(state_int=state_int)) + "\nq values: " + str(self.dqn(State(state_int).tensor_version()[np.newaxis, :])))
     
    def train(self, game):
        training_scores = []
        for epoch in range(self.epochs):
            score = game.play(training_mode=True, max_steps=1000)
            print("\nepoch: " + str(epoch))
            print("Score: " + str(score))
            print("exploration probability: " + str(self.exploration_prob))
            training_scores.append(score)
            game.reset()
            self.exploration_prob *= self.exploration_decay_rate

        print("epochs: " + str(self.epochs))
        print("learning rate: " + str(self.learning_rate))
        print("discount factor: " + str(self.discount_factor))
        print("exploration prob: " + str(self.exploration_prob))
        print("training scores: " + "\naverage: " + str(sum(training_scores) / len(training_scores)))
        print("max: " + str(max(training_scores)))
        print("Last 100 average: " + str(sum(training_scores[-100:]) / 100.0))
        print("\n\n")
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()
        return (sum(training_scores[-100:]) / 100.0, (self.epochs, self.learning_rate, self.discount_factor, self.exploration_prob))

    def tune_hyperparameters(self, game):
            # epochss = [1000, 5000, 10_000, 15_000, 20_000]
            # learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            # discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            # exploration_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            # epochss = [1000, 5000, 10_000]
            # learning_rates = [0.2, 0.4, 0.6, 0.8, 1]
            # discount_factors = [0.2, 0.4, 0.6, 0.8, 1]
            # exploration_probs = [0.9]
            # exploration_probs = [0.005, 0.01, 0.05, 0.1]

        epochss = [5_000]
        learning_rates = [0.001]
        discount_factors = [0.5, 0.8, 0.95]
        exploration_probs = [1.0]

        results = []
        for epochs in epochss:
            for learning_rate in learning_rates:
                for discount_factor in discount_factors:
                    for exploration_prob in exploration_probs:
                        self.epochs = epochs
                        self.learning_rate = learning_rate
                        self.discount_factor = discount_factor
                        self.exploration_prob = exploration_prob
                        results.append(self.train(game))
        print("results: " + str(results))
        print("Max: " + str(max(results)))
        plt.plot(discount_factor, results)
        plt.show()