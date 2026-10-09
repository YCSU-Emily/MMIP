import json
import sys
import time
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

ROOT = Path.home() / "MMIP"
W4_DIR = ROOT / "week4/quiz4/image_captioning"
W5_DIR = ROOT / "week5/quiz3"

INPUT_JSON = W5_DIR / "results/clip_lstm_test_captions.json"
IMAGE_DIR = W4_DIR / "data/images/test"
OUTPUT_DIR = W5_DIR / "results"

sys.path.insert(0, str(W4_DIR))
import inference as inf


def load_shared_filenames():
    rows = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    filenames = [row["file_name"] for row in rows]

    if len(filenames) != 100:
        raise ValueError(f"預期 100 張圖片，實際為 {len(filenames)} 張")

    if len(set(filenames)) != len(filenames):
        raise ValueError("Week 5 圖片清單有重複檔名")

    missing = [name for name in filenames if not (IMAGE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 張圖片不存在，例如：{missing[:5]}"
        )

    return filenames


def run_inference(label, checkpoint_path, output_path, filenames, transform, device):
    print("\n" + "=" * 72)
    print(f"模型：{label}")
    print(f"Checkpoint：{checkpoint_path}")
    print("=" * 72)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab = inf.Vocabulary()
    vocab.word2idx = dict(checkpoint["vocab_word2idx"])
    vocab.idx2word = {
        int(k): v for k, v in checkpoint["vocab_idx2word"].items()
    }

    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    max_length = checkpoint["max_length"]

    encoder = inf.EncoderCNN(embed_dim).to(device)
    decoder = inf.DecoderLSTM(
        embed_dim, hidden_dim, len(vocab)
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    results = []
    total_time = 0.0

    with torch.inference_mode():
        for i, filename in enumerate(filenames, start=1):
            image_path = IMAGE_DIR / filename
            with Image.open(image_path) as image:
                image_tensor = transform(image.convert("RGB"))

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            caption = inf.generate_caption(
                encoder,
                decoder,
                image_tensor,
                vocab,
                max_length,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_time += elapsed

            results.append({
                "file_name": filename,
                "generated_caption": caption,
                "inference_time_sec": elapsed,
            })

            if i == 1 or i % 10 == 0:
                print(f"[{i:03d}/100] {filename}")
                print(f"  Caption: {caption}")
                print(f"  Time: {elapsed * 1000:.2f} ms")

    output = {
        "model": "Week 4 ResNet18 + LSTM",
        "experiment": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_loss": checkpoint.get("val_loss"),
        "num_images": len(results),
        "total_inference_time_sec": total_time,
        "average_inference_time_ms": total_time / len(results) * 1000,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n已儲存：{output_path}")
    print(f"圖片數：{len(results)}")
    print(f"平均推論時間：{output['average_inference_time_ms']:.2f} ms/image")


def main():
    filenames = load_shared_filenames()
    print(f"確認共用圖片數：{len(filenames)}")
    print("前 5 張：", filenames[:5])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    experiments = [
        (
            "baseline",
            W4_DIR / "models/best_caption_model.pth",
            OUTPUT_DIR / "week4_baseline_shared100.json",
        ),
        (
            "continuation",
            W4_DIR / "models/experiments/continuation_v1_best.pth",
            OUTPUT_DIR / "week4_continuation_shared100.json",
        ),
    ]

    for label, checkpoint_path, output_path in experiments:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

        run_inference(
            label, checkpoint_path, output_path,
            filenames, transform, device,
        )

    print("\n兩個 Week 4 模型的共用圖片推論都已完成。")


if __name__ == "__main__":
    main()
