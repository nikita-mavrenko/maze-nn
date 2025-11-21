from dataset import DatasetGenerator
from maze import MazeGenerator


def main():
    print("Генерация датасета")
    dataset_generator = DatasetGenerator(maze_size=5, dataset_size=100)
    X, y, metadata = dataset_generator.generate_dataset()
    dataset_generator.save_dataset(X, y, metadata, "dataset.pkl")

    # загрузка из датасета
    X_loaded, y_loaded, metadata_loaded = dataset_generator.load_dataset("dataset.pkl")
    print(f"\nЗагружен датасет: {X_loaded.shape[0]} samples")

if __name__ == "__main__":
    main()