
# Snake_DQN — Aprendizaje por Refuerzo Profundo para Snake

Este proyecto implementa un agente de Deep Q-Learning (DQN) que aprende a jugar Snake desde cero, utilizando PyTorch para el entrenamiento, Pygame para la visualización y una arquitectura modular que facilita la experimentación, la claridad visual y la mantenibilidad.

Objetivos

- Entrenar un agente inteligente que aprenda a jugar Snake mediante aprendizaje por refuerzo
- Visualizar el progreso del entrenamiento en tiempo real (score, promedio y epsilon)
- Centralizar la configuración para facilitar pruebas, refactorizaciones y comparaciones
- Diseñar una arquitectura clara, modular y pedagógica para uso educativo
- Evaluar automáticamente todos los modelos guardados y detectar el mejor

Instalación

1) Clonar el repositorio

- git clone https://github.com/PaperClip01/Snake_DRL.git
- cd Snake_DQN

2) Crear entorno virtual

- python -m venv venv
- source venv/bin/activate      # Linux/Mac
- venv\Scripts\activate         # Windows

3) Instalar dependencias

- pip install -r requirements.txt

4) Ejecutar el proyecto

- python main.py
  
- Modos disponibles
- Al iniciar el programa, se te pedirá elegir entre:
- 1 → Entrenar modelo desde cero
- 2 → Continuar entrenamiento con modelo guardado
- 3 → Mostrar modelo guardado sin entrenar
- 4 → Evaluar todos los modelos guardados y mostrar el mejor

Detalles:
- Los modelos se guardan automáticamente solo si superan su score anterior
- El nombre del modelo incluye la fecha y el score: model_YYYY-MM-DD_score_X.pth
- La evaluación se realiza sin renderizado, a máxima velocidad
- El modo demo permite visualizar el comportamiento del modelo ganador

Estructura del proyecto
- Snake_DQN(V3.0)/
- main.py                         # Punto de entrada centralizado
- agent.py                        # Lógica del agente DQN y memoria
- model.py                        # Red neuronal y entrenador
- game.py                         # Entorno Snake con Pygame (renderizable o silencioso)
- metrics.py                      # Visualización de score, promedio y epsilon
- config.py                       # Parámetros globales centralizados
- model/model_IDX_score_X.pth     # Carpeta donde se guardan los modelos entrenados 

Archivos generados
- model/model_IDX_score_X.pth → Modelo entrenado con score máximo alcanzado
- Visualización en tiempo real del entrenamiento (no se guarda por defecto)

Créditos:

Este proyecto fue desarrollado por Manuel J Palavecino, como parte de la materia Inteligencia Artificial de la Universidad del CEMA. Podés usarlo, modificarlo y compartirlo libremente para fines educativos.






