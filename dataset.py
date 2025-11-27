import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from maze import MazeGenerator


class DatasetGenerator:
    def __init__(self, maze_size: int = 10, dataset_size: int = 1000):
        self.maze_size = maze_size
        self.dataset_size = dataset_size
        self.maze_generator = MazeGenerator(maze_size)
        self.label_encoder = LabelEncoder()

        # вверх, вниз, влево, вправо
        self.actions = ['W', 'S', 'A', 'D']
        self.label_encoder.fit(self.actions)

    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        X = []  # лабиринт + позиция
        y = []  # следующее действие
        metadata = []

        print(f"Генерация датасета из {self.dataset_size} лабиринтов...")

        for i in range(self.dataset_size):
            if (i + 1) % 100 == 0:
                print(f"Сгенерировано {i + 1}/{self.dataset_size} лабиринтов")

            maze_matrix, maze_obj = self.maze_generator.generate_maze()

            optimal_path = self.maze_generator.get_path()

            if optimal_path is None:
                continue

            maze_samples, action_samples, maze_meta = self._extract_training_samples(
                maze_matrix, optimal_path, maze_obj
            )

            X.extend(maze_samples)
            y.extend(action_samples)
            metadata.extend(maze_meta)

        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.int64)

        print(f"Датасет сгенерирован: {len(X_array)} samples")
        print(f"Размерность X: {X_array.shape}")
        print(f"Размерность y: {y_array.shape}")

        return X_array, y_array, metadata

    def _extract_training_samples(self, maze_matrix: np.ndarray,
                                  optimal_path: List[Tuple[int, int]],
                                  maze_obj: Any) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Извлекает тренировочные samples из оптимального пути"""
        samples_X = []
        samples_y = []
        samples_meta = []

        for i in range(len(optimal_path) - 1):
            current_pos = optimal_path[i]
            next_pos = optimal_path[i + 1]

            action = self._get_action(current_pos, next_pos)
            if action is None:
                continue

            state_representation = self._create_state_representation(
                maze_matrix, current_pos, maze_obj
            )

            samples_X.append(state_representation)
            samples_y.append(self.label_encoder.transform([action])[0])

            samples_meta.append({
                'current_pos': current_pos,
                'next_pos': next_pos,
                'action': action,
                'step': i,
                'maze_size': self.maze_size
            })

        return samples_X, samples_y, samples_meta

    def _get_action(self, current_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> str:
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        if dx == -1 and dy == 0:
            return 'W'
        elif dx == 1 and dy == 0:
            return 'S'
        elif dx == 0 and dy == -1:
            return 'A'
        elif dx == 0 and dy == 1:
            return 'D'
        else:
            return None

    def _create_state_representation(self, maze_matrix: np.ndarray,
                                     current_pos: Tuple[int, int],
                                     maze_obj: Any) -> np.ndarray:
        """
        - 0: свободная клетка
        - 1: стена
        - 2: текущая позиция
        - 3: целевая позиция
        """
        matrix_size = self.maze_size * 2 + 1
        state = np.zeros((matrix_size, matrix_size), dtype=np.float32)

        state[:] = maze_matrix

        agent_x, agent_y = self._convert_to_matrix_coords(current_pos)
        state[agent_x, agent_y] = 2

        goal_x, goal_y = self._convert_to_matrix_coords(self.maze_generator.goal)
        state[goal_x, goal_y] = 3

        state = state / 3.0

        return state

    def _convert_to_matrix_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        return (2 * pos[0] - 1, 2 * pos[1] - 1)

    def save_dataset(self, X: np.ndarray, y: np.ndarray, metadata: List[Dict],
                     filename: str = "maze_dataset.pkl"):
        dataset = {
            'X': X,
            'y': y,
            'metadata': metadata,
            'action_encoder': self.label_encoder,
            'maze_size': self.maze_size,
            'dataset_size': len(X)
        }

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Датасет сохранен в {filename}")

    def load_dataset(self, filename: str = "maze_dataset.pkl") -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)

        self.label_encoder = dataset['action_encoder']
        return dataset['X'], dataset['y'], dataset['metadata']
