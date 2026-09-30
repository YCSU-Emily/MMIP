import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 224

CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Local",
    "Near-Full",
    "Normal",
    "Random",
    "Scratch",
]


def get_transforms():
    """
    Quiz 2 baseline:
    不使用 Data Augmentation。
    Quiz 3 再加入 augmentation。
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return transform


def get_datasets(data_root):
    transform = get_transforms()

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4
):
    train_dataset, val_dataset, test_dataset = get_datasets(data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    )
