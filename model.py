import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import LR, GAMMA  # hiperparámetros centralizados

# Clase que define la arquitectura de la red neuronal utilizada por el agente
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)   # Capa oculta: transforma entrada en representación intermedia
        self.linear2 = nn.Linear(hidden_size, output_size)  # Capa de salida: produce Q-values para cada acción
 
    def forward(self, x):
        x = F.relu(self.linear1(x)) # Aplica ReLU para introducir no linealidad
        x = self.linear2(x)         # Calcula los Q-values finales sin activación
        return x                    # Devuelve los Q-values para cada acción posible

    def save(self, file_name='model.pth'): # Guarda los pesos actuales de la red neuronal en el archivo especificado
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name)) # Carga los pesos desde el archivo
        self.eval() # Pone la red en modo evaluación (desactiva dropout, batchnorm, etc.)

# Clase que encapsula el proceso de entrenamiento usando Q-learning
class QTrainer:
    def __init__(self, model, lr=LR, gamma=GAMMA):
        self.lr = lr                                                # Tasa de aprendizaje
        self.gamma = gamma                                          # Factor de descuento para recompensas futuras
        self.model = model                                          # Red neuronal que se va a entrenar
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Optimizador Adam
        self.criterion = nn.MSELoss()                               # Función de pérdida: error cuadrático medio

    
    def train_step(self, state, action, reward, next_state, done):
        # Convierte las entradas en tensores de PyTorch
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Si se trata de una sola transición, se expande la dimensión para uniformidad
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # Convierte el booleano en tupla para iteración

        pred = self.model(state) # Predicción actual de Q-values
        target = pred.clone()    # Clona la predicción para construir el objetivo corregido

        # Itera sobre cada transición del lote
        for idx in range(len(done)):
            Q_new = reward[idx] # Valor inmediato de recompensa
            if not done[idx]:
                # Si el episodio no terminó, se suma el valor futuro descontado
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) # Bellman 
            # Actualiza el Q-value correspondiente a la acción tomada
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()          # Reinicia los gradientes acumulados
        loss = self.criterion(target, pred) # Calcula la pérdida entre predicción y objetivo
        loss.backward()                     # Propagación hacia atrás para calcular gradientes
        self.optimizer.step()               # Actualiza los pesos del modelo
