# Hiperparámetros del agente
MAX_MEMORY = 100_000    # Tamaño máximo de la memoria de experiencia (deque). Cuanto mayor sea, más diversidad de experiencias puede aprender el agente.
BATCH_SIZE = 1000       # Cantidad de transiciones que se seleccionan aleatoriamente de la memoria para entrenar en cada paso.
LR = 0.001              # Tasa de aprendizaje del optimizador Adam. Controla qué tan rápido se actualizan los pesos de la red neuronal. Un valor bajo permite ajustes más finos y estables.
GAMMA = 0.9             # Factor de descuento para recompensas futuras. Determina cuánto valora el agente las recompensas a largo plazo frente a las inmediatas. Un valor cercano a 1 favorece decisiones estratégicas.
EPSILON_START = 80      # Valor inicial de epsilon, que controla la probabilidad de exploración aleatoria. Al comenzar, el agente explora mucho para recolectar experiencias diversas.
MIN_EPSILON = 5         # Valor mínimo de epsilon. El agente siempre tendrá al menos un 2,5% de probabilidad de tomar decisiones aleatorias (5/200), lo que evita el sobreajuste y mantiene algo de exploración.

# Parámetros del entorno
BLOCK_SIZE = 20 # Tamaño de cada bloque de la serpiente y la comida en píxeles.
SPEED = 40      # Velocidad de actualización del juego (FPS).
WIDTH = 640     # Ancho de la ventana del juego en píxeles. 
HEIGHT = 480    # Alto de la ventana del juego en píxeles. 

# Arquitectura del modelo
INPUT_SIZE = 11     # Dimensión del vector de estado que recibe la red neuronal. Representa información como peligros, dirección actual y ubicación de la comida.
HIDDEN_SIZE = 256   # Cantidad de neuronas en la capa oculta de la red. Más neuronas permiten aprender patrones más complejos.
OUTPUT_SIZE = 3     # Cantidad de acciones posibles que puede tomar el agente: [seguir recto, girar derecha, girar izquierda]. 


# Parámetros de entrenamiento y evaluación
MODEL_DIR = 'model'
NUM_EVAL_EPISODES = 20
RENDER_EVAL = False
RENDER_TRAINING = True
RENDER_DEMO = True

# Visualización de métricas
PLOT_TITLE = "Training Progress"
PLOT_XLABEL = "Number of Games"
PLOT_YLABEL_SCORE = "Score"
PLOT_YLABEL_EPSILON = "Epsilon"

COLOR_SCORE = "blue"
COLOR_MEAN = "orange"
COLOR_EPSILON = "gray"
LINESTYLE_EPSILON = "--"

EPSILON_YMAX = 100
PLOT_PAUSE = 0.1






