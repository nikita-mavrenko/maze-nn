import gc
import os
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from maze import MazeGenerator


class DatasetGenerator:
    def __init__(self, maze_size: int = 10, dataset_size: int = 1000):
        self.maze_size = maze_size
        self.dataset_size = dataset_size
        self.maze_generator = MazeGenerator(maze_size)
        self.label_encoder = LabelEncoder()

        # вверх, вниз, влево, вправо
        self.label_encoder.fit(['N', 'S', 'W', 'E'])

    def generate_dataset(self, chunks_folder, chunks=10):
        chunk_size = self.dataset_size // chunks

        chunks_folder = f"dataset_chunks_{self.maze_size}x{self.maze_size}"
        os.makedirs(chunks_folder, exist_ok=True)
        all_chunks = []

        print(f"Генерация датасета длиной {self.dataset_size} по {chunks} чанков")

        for chunk_idx in range(chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, self.dataset_size)

            X_chunk, y_chunk, meta_chunk = [], [], []

            print(f"Генерируем чанк {chunk_idx + 1}/{chunks} ({end_idx - start_idx} лабиринтов)...")

            for i in range(start_idx, end_idx):
                maze_matrix, maze_obj = self.maze_generator.generate_maze()
                path = self.maze_generator.get_path()
                if not path or len(path) < 2:
                    continue

                samples_X, samples_y, samples_meta = self._extract_training_samples(
                    maze_matrix, path, maze_obj
                )
                X_chunk.extend(samples_X)
                y_chunk.extend(samples_y)
                meta_chunk.extend(samples_meta)
                print(f"Сгенерирован лабиринт: {i - start_idx + 1}/{end_idx - start_idx}")

            chunk_path = f"{chunks_folder}/chunk_{chunk_idx:03d}.npz"
            np.savez_compressed(
                chunk_path,
                X=np.array(X_chunk, dtype=np.float32),
                y=np.array(y_chunk, dtype=np.int64),
                metadata=meta_chunk
            )
            print(f"Чанк сохранён: {chunk_path} ({len(X_chunk)} примеров)")
            all_chunks.append(chunk_path)

            del X_chunk, y_chunk, meta_chunk
            gc.collect()

        print("Генерация завершена! Чанки готовы.")
        return all_chunks

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

            state_representation = self.create_state_representation(
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

    def _get_action(self, current_pos, next_pos):
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        if dx == -1 and dy == 0: return 'N'  # вверх
        if dx == 1 and dy == 0: return 'S'  # вниз
        if dx == 0 and dy == -1: return 'W'  # влево
        if dx == 0 and dy == 1: return 'E'  # вправо
        return None

    def create_state_representation(self, maze_matrix: np.ndarray,
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


class ChunkedMazeDataset(torch.utils.data.Dataset):
    def __init__(self, chunk_files):
        self.chunk_files = chunk_files
        self.lengths = []
        self.cumsum = [0]
        for f in chunk_files:
            data = np.load(f)
            length = len(data['y'])
            self.lengths.append(length)
            self.cumsum.append(self.cumsum[-1] + length)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        chunk_idx = next(i for i, cum in enumerate(self.cumsum) if idx < cum) - 1
        local_idx = idx - self.cumsum[chunk_idx]
        data = np.load(self.chunk_files[chunk_idx])
        return torch.from_numpy(data['X'][local_idx]), torch.tensor(int(data['y'][local_idx]))
