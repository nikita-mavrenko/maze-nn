import os
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
import torch
import glob

from torch.utils.data import TensorDataset, DataLoader

from dataset import DatasetGenerator, ChunkedMazeDataset
from nn import NeuralNetwork
from trainer import NNTrainer


MAZE_SIZE = 10
DATASET_SIZE = 100
EPOCHS = 50
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

if __name__ == "__main__":
    main()