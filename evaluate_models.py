import os
from agent import Agent
from game import SnakeGameAI
from config import MODEL_DIR, NUM_EVAL_EPISODES, RENDER_EVAL

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]

def evaluate_model(model_path, num_episodes=NUM_EVAL_EPISODES):
    agent = Agent()
    game = SnakeGameAI(render=RENDER_EVAL)
    agent.model.load(os.path.join(MODEL_DIR, model_path))

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
    avg_score = total_score / num_episodes
    return avg_score

def main():
    models = list_models()
    if not models:
        print(f"No hay modelos guardados en la carpeta '{MODEL_DIR}/'.")
        return

    print(f"Evaluando {len(models)} modelos por {NUM_EVAL_EPISODES} partidas cada uno...\n")
    results = []
    for model_name in models:
        avg = evaluate_model(model_name)
        results.append((model_name, avg))
        print(f"Modelo: {model_name}: promedio {avg:.2f}")

    results.sort(key=lambda x: x[1], reverse=True)
    best_model, best_score = results[0]
    print("\nMejor modelo:")
    print(f"{best_model} → promedio {best_score:.2f}")

if __name__ == '__main__':
    main()