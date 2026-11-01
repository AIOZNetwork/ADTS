from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(train_path: str, test_path: str, batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    train = ImageFolder(train_path, transform=transform)
    test = ImageFolder(test_path, transform=transform)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size)
    )
