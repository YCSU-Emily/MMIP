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


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(augment=False):
    """
    Quiz 3 Data Augmentation

    augment=False:
        使用原本 Quiz 2 baseline preprocessing。

    augment=True:
        僅對 training data 加入 Data Augmentation。
    """

    if augment:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

            # 幾何變化
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),

            # 輕微影像強度變化
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=MEAN,
                std=STD
            ),
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD
            ),
        ])

    return transform


def get_datasets(data_root, augment=False):
    """
    建立 Train / Validation / Test Dataset。

    只有 Training Dataset 可以使用 augmentation。
    Validation / Test 永遠使用固定 preprocessing，
    確保評估結果公平。
    """

    train_transform = get_transforms(
        augment=augment
    )

    eval_transform = get_transforms(
        augment=False
    )

    train_dir = os.path.join(
        data_root,
        "train"
    )

    val_dir = os.path.join(
        data_root,
        "val"
    )

    test_dir = os.path.join(
        data_root,
        "test"
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


def get_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4,
    augment=False
):
    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = get_datasets(
        data_root,
        augment=augment
    )

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
