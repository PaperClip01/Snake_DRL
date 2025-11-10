from agent import Agent
from game import SnakeGameAI
from metrics import plot
from config import MODEL_DIR, NUM_EVAL_EPISODES, RENDER_TRAINING, RENDER_EVAL, RENDER_DEMO
import pygame
import os

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')] if os.path.exists(MODEL_DIR) else []

def select_model():
    models = list_models()
    if not models:
        print("No hay modelos guardados.")
        return None
    print("\nModelos disponibles:")
    for i, name in enumerate(models):
        print(f"{i + 1}: {name}")
    while True:
        try:
            idx = int(input("Seleccioná el número del modelo: ")) - 1
            if 0 <= idx < len(models):
                return os.path.join(MODEL_DIR, models[idx])
        except ValueError:
            pass
        print("Selección inválida. Intentá de nuevo.")

def get_next_model_id():
    existing = [f for f in list_models() if f.startswith("model_ID")]
    ids = []
    for f in existing:
        try:
            id_part = f.split('_')[1]
            if id_part.startswith("ID") and id_part[2:].isdigit():
                ids.append(int(id_part[2:]))
        except:
            continue
    next_id = max(ids) + 1 if ids else 1
    return next_id

def generate_model_name(model_id, score):
    return f"model_ID{model_id}_score_{score}.pth"

def get_model_path_by_id(model_id):
    prefix = f"model_ID{model_id}_score_"
    for f in list_models():
        if f.startswith(prefix):
            return os.path.join(MODEL_DIR, f)
    return None

def evaluate_model(model_path, num_episodes=NUM_EVAL_EPISODES):
    agent = Agent()
    game = SnakeGameAI(render=RENDER_EVAL)
    agent.load(model_path)

    total_score = 0
    for _ in range(num_episodes):
        while True:
            state = agent.get_state(game)
            final_move = agent.get_action(state)
            reward, done, score = game.play_step(final_move)
            if done:
                total_score += score
                game.reset()
                break
    return total_score / num_episodes

def run_demo(model_path):
    agent = Agent()
    game = SnakeGameAI(render=RENDER_DEMO)
    agent.load(model_path)
    print(f"Ejecutando modelo en modo demo: {model_path}")
    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)
        if done:
            game._update_ui()
            pygame.time.delay(1000)
            game.reset()
            print(f'Demo , Score: {score}')

def main():
    print("Seleccioná una opción:")
    print("1 → Entrenar modelo desde cero")
    print("2 → Continuar entrenamiento con modelo guardado")
    print("3 → Mostrar modelo guardado sin entrenar")
    print("4 → Evaluar todos los modelos guardados y mostrar el mejor")
    modo = input("Opción (1/2/3/4): ")

    plot_scores = []
    plot_mean_scores = []
    epsilon_values = []
    total_score = 0
    record = 0
    model_path = None
    model_id = None

    if modo == '4':
        models = list_models()
        if not models:
            print("No hay modelos guardados.")
            return
        print(f"\nEvaluando {len(models)} modelos por {NUM_EVAL_EPISODES} partidas cada uno...\n")
        results = []
        for model_name in models:
            path = os.path.join(MODEL_DIR, model_name)
            avg = evaluate_model(path)
            results.append((model_name, avg))
            print(f"modelo: {model_name}: promedio {avg:.2f}")
        results.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score = results[0]
        best_path = os.path.join(MODEL_DIR, best_model)
        print(f"\nMejor modelo: {best_model} → promedio {best_score:.2f}")
        run = input("\n¿Deseás correrlo en modo demo? (s/n): ").lower()
        if run == 's':
            run_demo(best_path)
        return

    agent = Agent()
    game = SnakeGameAI(render=RENDER_TRAINING)

    if modo == '1':
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_id = get_next_model_id()
        model_path = os.path.join(MODEL_DIR, generate_model_name(model_id, 0))
        print(f"Entrenando nuevo modelo: {model_path}")

    elif modo == '2':
        model_path = select_model()
        if model_path:
            agent.load(model_path)
            print(f"Modelo cargado: {model_path}")
            try:
                model_id = int(model_path.split('_')[1][2:])
                record = int(model_path.split('_score_')[-1].split('.')[0])
            except:
                model_id = None
                record = 0
        else:
            print("No se seleccionó ningún modelo. Saliendo.")
            return

    elif modo == '3':
        model_path = select_model()
        if model_path:
            run_demo(model_path)
        else:
            print("No se seleccionó ningún modelo. Saliendo.")
        return

    # Entrenamiento (modo 1 o 2)
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            if RENDER_TRAINING:
                game._update_ui()
                pygame.time.delay(500)

            agent.n_games += 1
            agent.train_long_memory()
            game.reset()

            if score > record:
                record = score
                old_path = get_model_path_by_id(model_id)
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                model_path = os.path.join(MODEL_DIR, generate_model_name(model_id, record))
                agent.save(model_path)
                print(f'Nuevo récord: {record} → modelo actualizado como {model_path}')

            print(f'Game {agent.n_games} , Score: {score} , Record: {record} , Epsilon: {agent.epsilon}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            epsilon_values.append(agent.epsilon)

            plot(plot_scores, plot_mean_scores, epsilon_values)

if __name__ == '__main__':
    main()