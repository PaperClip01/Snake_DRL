import torch 
import random  
import numpy as np  
from collections import deque 
from game import Direction, Point 
from model import Linear_QNet, QTrainer 
from config import MAX_MEMORY, BATCH_SIZE, LR, GAMMA, EPSILON_START, MIN_EPSILON  # Hiperparámetros desde config
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE  # Tamaños de la red neuronal

class Agent:
    # Clase que representa al agente inteligente que aprende a jugar Snake
    def __init__(self):
        self.n_games = 0                                                # Contador de partidas jugadas
        self.epsilon = EPSILON_START                                    # Nivel de exploración inicial
        self.gamma = GAMMA                                              # Factor de descuento para Q-Learning
        self.memory = deque(maxlen=MAX_MEMORY)                          # Memoria de experiencias (experiencia reciente se mantiene)
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # Red neuronal que predice Q-values
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)    # Entrenador que ajusta los pesos de la red

    def get_state(self, game):
        # Extrae el estado actual del entorno como vector binario (entrada para la red)
        head = game.snake[0]  # Cabeza de la serpiente

        # Puntos adyacentes a la cabeza en las 4 direcciones
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

        # Dirección actual de movimiento
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Vector de estado: peligros, dirección actual y ubicación relativa de la comida
        state = [
            # Peligro al frente
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Peligro a la derecha
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Peligro a la izquierda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Dirección actual de movimiento (one-hot)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Ubicación relativa de la comida
            game.food.x < game.head.x,  # Comida a la izquierda
            game.food.x > game.head.x,  # Comida a la derecha
            game.food.y < game.head.y,  # Comida arriba
            game.food.y > game.head.y   # Comida abajo
        ]

        return np.array(state, dtype=int)  # Devuelve el estado como array binario

    def remember(self, state, action, reward, next_state, done):
        # Almacena una transición en la memoria para entrenamiento posterior
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Entrenamiento por lotes a partir de la memoria (replay)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Muestra aleatoria
        else:
            mini_sample = self.memory  # Usa toda la memoria si es pequeña

        # Desempaqueta las transiciones
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Entrenamiento inmediato con una sola transición (útil en tiempo real)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decide la acción a tomar: exploración vs explotación
        self.epsilon = max(MIN_EPSILON, EPSILON_START - self.n_games)  # Decae con el tiempo
        final_move = [0, 0, 0]  # Representación one-hot de la acción

        if random.randint(0, 200) < self.epsilon:
            # Exploración: elige una acción aleatoria
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Explotación: elige la mejor acción según la red
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Q-values
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move  # Acción codificada como [1,0,0], [0,1,0] o [0,0,1]

    def save(self, file_name):
        # Guarda el modelo y el estado del agente (incluye epsilon y n_games)
        torch.save({
            'model_state': self.model.state_dict(),
            'epsilon': self.epsilon,
            'n_games': self.n_games
        }, file_name)

    def load(self, file_name):
        # Carga el modelo y el estado del agente desde un archivo
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()  # Modo evaluación (desactiva dropout, etc.)
        self.epsilon = checkpoint.get('epsilon', EPSILON_START)  # Restaura epsilon
        self.n_games = checkpoint.get('n_games', 0)  # Restaura contador de partidas