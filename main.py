import os

os.environ["TORCH_DYNAMO_DISABLE"] = "1"
import numpy as np
from maze import MazeGenerator

import torch
import glob

from torch.utils.data import DataLoader

from dataset import DatasetGenerator, ChunkedMazeDataset
from nn import NeuralNetwork
from trainer import NNTrainer

MAZE_SIZE = 10
DATASET_SIZE = 200
EPOCHS = 20
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 64
CHUNKS = 20

DATASET_FOLDER = f"dataset_chunks_{MAZE_SIZE}x{MAZE_SIZE}"
MODEL_FILENAME = f"maze_solver_{MAZE_SIZE}x{MAZE_SIZE}.pth"


def main():
    dataset_generator = DatasetGenerator(maze_size=MAZE_SIZE, dataset_size=DATASET_SIZE)

    X, y, metadata = None, None, None

    chunk_files = None
    if not os.path.exists(DATASET_FOLDER):
        print(f"Датасет не найден, генерируем новый: {DATASET_FOLDER}")
        chunk_files = dataset_generator.generate_dataset(chunks=20, chunks_folder=DATASET_FOLDER)
    else:
        print(f"Датасет найден: {DATASET_FOLDER}")
        chunk_files = sorted(glob.glob(f"{DATASET_FOLDER}/chunk_*.npz"))

    train_model = True
    if os.path.exists(MODEL_FILENAME):
        print(f"Найдена обученная модель {MODEL_FILENAME}")
        train_model = False
    else:
        print("Обученная модель не найдена")

    dataset = ChunkedMazeDataset(chunk_files)

    test_size = int(len(dataset) * TEST_SPLIT)
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size - test_size

    train_ds, validation_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Тренировочных батчей: {len(train_loader)}")
    print(f"Валидационных батчей: {len(validation_loader)}\n")
    print(f"Тестовых батчей: {len(test_loader)}\n")

    input_shape = (MAZE_SIZE * 2 + 1, MAZE_SIZE * 2 + 1)
    model = NeuralNetwork(input_shape=input_shape, num_actions=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not train_model:
        print(f"Загружаем веса из {MODEL_FILENAME}")
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device('cpu')))
        model.eval()
        print(f"Модель загружена")
    else:
        print(f"Запускаем обучение")
        trainer = NNTrainer(model)
        trainer.device = device
        trainer.train(train_loader, validation_loader, epochs=EPOCHS)
        torch.save(model.state_dict(), MODEL_FILENAME)
        print(f"Модель сохранена {MODEL_FILENAME}")

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = torch.nn.functional.cross_entropy(outputs, batch_y)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    test_accuracy = 100 * correct / total
    test_loss = test_loss / len(test_loader)

    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})")
    print(f"Размер тестовой выборки: {len(test_ds)} примеров")

    test_single_maze(model, device)


def solve_maze_with_nn(model, maze_matrix, maze_generator, device):
    current_pos = maze_generator.start
    goal = maze_generator.goal
    path = [current_pos]
    visited = {current_pos}
    max_steps = MAZE_SIZE * MAZE_SIZE * 3

    step = 0

    while current_pos != goal and step < max_steps:
        state = create_state_for_nn(maze_matrix, current_pos, maze_generator)
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = model(state_tensor)
            action_probs = torch.softmax(outputs, dim=1)[0]

        possible_moves = get_possible_moves(current_pos, maze_generator)

        moved = False
        for action_idx in torch.argsort(outputs, descending=True)[0]:
            action = index_to_action(action_idx.item())
            next_pos = get_next_position(current_pos, action)

            if next_pos and next_pos in possible_moves and next_pos not in path[-5:]:
                current_pos = next_pos
                if current_pos not in visited:
                    path.append(current_pos)
                    visited.add(current_pos)
                moved = True
                step += 1
                print(f"Шаг {step}: {current_pos}, действие: {action}, расстояние: {dist(current_pos, goal)}")
                break

        if not moved and possible_moves:
            print(f"Тупик на {current_pos}! Ищем выход...")

            found_exit = False
            for check_pos in reversed(path[-10:]):
                possible_here = get_possible_moves(check_pos, maze_generator)
                for move in possible_here:
                    if move not in path[-8:]:  # избегаем коротких циклов
                        current_pos = move
                        path.append(current_pos)
                        step += 1
                        print(f"Шаг {step}: найден выход → {current_pos} (backtrack)")
                        found_exit = True
                        break
                if found_exit:
                    break

    if current_pos == goal:
        print(f"Лабиринт пройден. Достигли цели за {len(path)} шагов")
        return path
    else:
        print(f"Не удалось пройти. Последняя позиция: {current_pos}")
        return path


def dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def create_state_for_nn(maze_matrix, current_pos, maze_generator):
    matrix_size = MAZE_SIZE * 2 + 1
    state = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    state[:] = maze_matrix

    # Отмечаем текущую позицию и цель
    ax, ay = (2 * current_pos[0] - 1, 2 * current_pos[1] - 1)
    gx, gy = (2 * maze_generator.goal[0] - 1, 2 * maze_generator.goal[1] - 1)

    state[ax, ay] = 2
    state[gx, gy] = 3

    return state / 3.0


def index_to_action(index):
    return ['N', 'S', 'W', 'E'][index]


def get_next_position(current_pos, action):
    x, y = current_pos
    if action == 'N': return (x - 1, y)
    if action == 'S': return (x + 1, y)
    if action == 'W': return (x, y - 1)
    if action == 'E': return (x, y + 1)
    return None


def get_possible_moves(current_pos, maze_generator):
    x, y = current_pos
    cell = (x, y)
    moves = []

    if maze_generator.maze_obj.maze_map[cell]['N']: moves.append((x - 1, y))
    if maze_generator.maze_obj.maze_map[cell]['S']: moves.append((x + 1, y))
    if maze_generator.maze_obj.maze_map[cell]['W']: moves.append((x, y - 1))
    if maze_generator.maze_obj.maze_map[cell]['E']: moves.append((x, y + 1))

    # Фильтруем только допустимые координаты
    moves = [(nx, ny) for nx, ny in moves if 1 <= nx <= MAZE_SIZE and 1 <= ny <= MAZE_SIZE]
    return moves


def test_single_maze(model, device):
    print("ТЕСТИРОВАНИЕ ЛАБИРИНТА")

    maze_generator = MazeGenerator(MAZE_SIZE)
    maze_matrix, maze_obj = maze_generator.generate_maze()

    optimal_path = maze_generator.get_path()
    if not optimal_path:
        print("Ошибка: оптимальный путь не найден")
        return

    print(f"Оптимальный путь: {len(optimal_path)} шагов")

    nn_path = solve_maze_with_nn(model, maze_matrix, maze_generator, device)

    if nn_path and nn_path[-1] == maze_generator.goal:
        efficiency = len(optimal_path) / len(nn_path) * 100
        print(f"Результат: {len(nn_path)} шагов ({efficiency:.1f}% от оптимального)")
        maze_generator.visualize_maze(path=nn_path, show_path=True)
    else:
        print(f"Нейросеть не достигла цели. Пройдено {len(nn_path) if nn_path else 0} шагов")
        if nn_path:
            maze_generator.visualize_maze(path=nn_path, show_path=True)
        else:
            maze_generator.visualize_maze(path=None, show_path=False)


if __name__ == "__main__":
    main()
