import random
from typing import Any
import pyamaze
import numpy as np
import time
from numpy import ndarray, dtype
from pyamaze import maze
from collections import deque


class MazeGenerator:
    def __init__(self, size):
        self.size = size
        self.maze_obj = None
        self.maze_matrix = None
        self.start = (size, size)
        self.goal = (1, 1)

    def generate_maze(self) -> tuple[ndarray[tuple[int, int], dtype[Any]], maze]:
        random.seed(int(time.time() * 1000) % 1000000)

        self.maze_obj = pyamaze.maze(self.size, self.size)
        self.maze_obj.CreateMaze(loopPercent=0)

        self.maze_matrix = self._convert_to_matrix()

        return self.maze_matrix, self.maze_obj

    def _convert_to_matrix(self) -> ndarray[tuple[int, int], dtype[Any]]:
        if self.maze_obj is None:
            raise ValueError("maze is not generated")

        matrix_size = self.size * 2 + 1
        matrix = np.ones((matrix_size, matrix_size), dtype=int)

        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                cell = (i, j)
                matrix[2 * i - 1][2 * j - 1] = 0

                if self.maze_obj.maze_map[cell]['E'] and 2 * j < matrix_size - 1:
                    matrix[2 * i - 1][2 * j] = 0
                if self.maze_obj.maze_map[cell]['W'] and 2 * j - 2 >= 0:
                    matrix[2 * i - 1][2 * j - 2] = 0
                if self.maze_obj.maze_map[cell]['N'] and 2 * i - 2 >= 0:
                    matrix[2 * i - 2][2 * j - 1] = 0
                if self.maze_obj.maze_map[cell]['S'] and 2 * i < matrix_size - 1:
                    matrix[2 * i][2 * j - 1] = 0

        return matrix

    def visualize_maze(self, path=None, show_path=True):
        if self.maze_obj is None:
            raise ValueError("maze is not generated")

        # Создаем агента для пути
        if path and show_path:
            a = pyamaze.agent(self.maze_obj, footprints=True)
            self.maze_obj.tracePath({a: path}, delay=0)

        self.maze_obj.run()

    def get_available_cells(self):
        return [(i, j) for i in range(1, self.size + 1) for j in range(1, self.size + 1)]

    def get_path(self, start=None, goal=None):
        return self._bfs_search(self.start, self.goal)

    def _bfs_search(self, start, goal):
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == goal:
                return path

            cell = (x, y)
            moves = []
            if self.maze_obj.maze_map[cell]['N']:
                moves.append((x - 1, y))
            if self.maze_obj.maze_map[cell]['S']:
                moves.append((x + 1, y))
            if self.maze_obj.maze_map[cell]['W']:
                moves.append((x, y - 1))
            if self.maze_obj.maze_map[cell]['E']:
                moves.append((x, y + 1))

            for move in moves:
                if (move not in visited and
                        1 <= move[0] <= self.size and
                        1 <= move[1] <= self.size):
                    visited.add(move)
                    queue.append((move, path + [move]))

        return None