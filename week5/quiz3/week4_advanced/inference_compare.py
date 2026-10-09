import os
import json
import re
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image


# ============================================================
# Configuration
# ============================================================

SEED = 3407

TEST_IMAGE_DIR = "./data/images/test"
TEST_CAPTION_JSON = "./data/captions/test.json"

CHECKPOINT = os.environ.get("CAPTION_CHECKPOINT", "./models/best_caption_model.pth")
OUTPUT_JSON = os.environ.get("CAPTION_OUTPUT", "./results/generated_captions.json")

IMAGE_SIZE = 224

NUM_IMAGES = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Tokenizer
# ============================================================

def tokenize(text):
    text = text.lower()

    return re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text
    )


# ============================================================
# Vocabulary
# ============================================================

class Vocabulary:

    def __init__(self, min_freq=5):

        self.min_freq = min_freq

        self.word2idx = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>",
        }

    def build(self, captions):

        frequency = {}

        for caption in captions:

            for word in tokenize(caption):

                frequency[word] = (
                    frequency.get(word, 0) + 1
                )

        for word, count in sorted(
            frequency.items()
        ):

            if count >= self.min_freq:

                idx = len(self.word2idx)

                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def __len__(self):

        return len(self.word2idx)

    def __getitem__(self, word):

        return self.word2idx.get(
            word,
            self.word2idx["<UNK>"]
        )


# ============================================================
# Load JSON
# ============================================================

def load_caption_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# Dataset
# ============================================================

class CaptionDataset(Dataset):

    def __init__(
        self,
        image_dir,
        coco_data,
        transform=None
    ):

        self.image_dir = image_dir
        self.transform = transform

        # ----------------------------------------------------
        # COCO format:
        #
        # images:
        #   {"id": 172330, "file_name": "000000172330.jpg"}
        #
        # annotations:
        #   {"image_id": 172330, "caption": "..."}
        # ----------------------------------------------------

        image_id_to_filename = {
            item["id"]: item["file_name"]
            for item in coco_data["images"]
        }

        # Keep one entry per image.
        self.image_files = []

        for image_id in sorted(
            image_id_to_filename.keys()
        ):

            filename = image_id_to_filename[
                image_id
            ]

            self.image_files.append(
                filename
            )

        self.image_id_to_filename = (
            image_id_to_filename
        )

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        filename = self.image_files[idx]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, filename


# ============================================================
# Encoder
#
# IMPORTANT:
# This structure matches train_caption.py.
# The trained checkpoint uses "resnet.*", not "backbone.*".
# ============================================================

class EncoderCNN(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        modules = list(
            backbone.children()
        )[:-1]

        # IMPORTANT: keep the name "resnet"
        # because the checkpoint contains resnet.*
        self.resnet = nn.Sequential(
            *modules
        )

        for param in self.resnet.parameters():

            param.requires_grad = False

        self.fc = nn.Linear(
            512,
            embed_dim
        )

    def forward(self, images):

        with torch.no_grad():

            features = self.resnet(
                images
            )

        features = features.view(
            features.size(0),
            -1
        )

        features = self.fc(
            features
        )

        return features


# ============================================================
# Decoder
# ============================================================

class DecoderLSTM(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.init_h = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.init_c = nn.Linear(
            embed_dim,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def init_hidden(self, features):

        h = self.init_h(
            features
        )

        c = self.init_c(
            features
        )

        h = h.unsqueeze(0)
        c = c.unsqueeze(0)

        return h, c

    def step(
        self,
        word,
        hidden
    ):

        embedding = self.embedding(
            word
        )

        output, hidden = self.lstm(
            embedding,
            hidden
        )

        output = self.fc(
            output
        )

        return output, hidden


# ============================================================
# Greedy Search
# ============================================================

@torch.no_grad()
def generate_caption(
    encoder,
    decoder,
    image,
    vocab,
    max_length
):

    image = image.unsqueeze(0).to(
        DEVICE
    )

    features = encoder(
        image
    )

    hidden = decoder.init_hidden(
        features
    )

    current_word = torch.tensor(
        [vocab["<START>"]],
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    generated_words = []

    for _ in range(max_length):

        output, hidden = decoder.step(
            current_word,
            hidden
        )

        next_word = output.argmax(
            dim=-1
        )

        word_id = next_word.item()

        word = vocab.idx2word.get(
            word_id,
            "<UNK>"
        )

        if word == "<END>":

            break

        if word not in [
            "<PAD>",
            "<START>",
            "<UNK>"
        ]:

            generated_words.append(
                word
            )

        current_word = next_word

    return " ".join(
        generated_words
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Week 4 Quiz 4 - "
        "Image Captioning Inference"
    )
    print("=" * 70)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    print()

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Restore vocabulary
    # --------------------------------------------------------

    vocab = Vocabulary()

    vocab.word2idx = dict(
        checkpoint["vocab_word2idx"]
    )

    vocab.idx2word = {
        int(k): v
        for k, v in checkpoint[
            "vocab_idx2word"
        ].items()
    }

    print(
        f"Vocabulary size: "
        f"{len(vocab):,}"
    )

    # --------------------------------------------------------
    # Restore configuration
    # --------------------------------------------------------

    embed_dim = checkpoint[
        "embed_dim"
    ]

    hidden_dim = checkpoint[
        "hidden_dim"
    ]

    max_length = checkpoint[
        "max_length"
    ]

    print(
        f"Embedding dimension: "
        f"{embed_dim}"
    )

    print(
        f"Hidden dimension: "
        f"{hidden_dim}"
    )

    print(
        f"Max caption length: "
        f"{max_length}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    print()
    print("Building model...")

    encoder = EncoderCNN(
        embed_dim
    ).to(DEVICE)

    decoder = DecoderLSTM(
        embed_dim,
        hidden_dim,
        len(vocab)
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    encoder.load_state_dict(
        checkpoint[
            "encoder_state_dict"
        ]
    )

    decoder.load_state_dict(
        checkpoint[
            "decoder_state_dict"
        ]
    )

    encoder.eval()
    decoder.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    test_data = load_caption_json(
        TEST_CAPTION_JSON
    )

    transform = transforms.Compose([
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    dataset = CaptionDataset(
        TEST_IMAGE_DIR,
        test_data,
        transform=transform
    )

    print(
        f"Test images: "
        f"{len(dataset):,}"
    )

    num_images = min(
        NUM_IMAGES,
        len(dataset)
    )

    # --------------------------------------------------------
    # Generate captions
    # --------------------------------------------------------

    print()
    print(
        f"Generating captions for "
        f"{num_images} images..."
    )

    results = []

    total_time = 0.0

    for i in range(num_images):

        image, filename = dataset[i]

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        caption = generate_caption(
            encoder,
            decoder,
            image,
            vocab,
            max_length
        )

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        results.append({
            "image": filename,
            "generated_caption": caption,
            "inference_time_sec": elapsed
        })

        if (
            (i + 1) % 10 == 0
            or i == 0
        ):

            print(
                f"[{i + 1:03d}/"
                f"{num_images:03d}] "
                f"{filename}"
            )

            print(
                f"    Caption: {caption}"
            )

            print(
                f"    Time: "
                f"{elapsed * 1000:.2f} ms"
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_time = (
        total_time / num_images
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    output = {
        "model": "ResNet18 + LSTM",
        "decoding": "Greedy Search",
        "num_images": num_images,
        "total_inference_time_sec": total_time,
        "average_inference_time_sec": average_time,
        "average_inference_time_ms":
            average_time * 1000,
        "checkpoint": CHECKPOINT,
        "best_epoch": checkpoint[
            "epoch"
        ],
        "validation_loss": checkpoint[
            "val_loss"
        ],
        "results": results
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Inference Complete")
    print("=" * 70)

    print(
        f"Images: {num_images}"
    )

    print(
        f"Total inference time: "
        f"{total_time:.4f} sec"
    )

    print(
        f"Average inference time: "
        f"{average_time * 1000:.4f} ms/image"
    )

    print(
        f"Results: "
        f"{OUTPUT_JSON}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
