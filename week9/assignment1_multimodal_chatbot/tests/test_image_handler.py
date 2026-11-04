import unittest
import tempfile

from pathlib import Path
from PIL import Image

from src.image_handler import analyze_image


class TestImageHandler(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_png(self):
        image_path = self.test_dir / "sample.png"

        Image.new("RGB", (640, 480)).save(image_path)

        result = analyze_image(image_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["image_format"], "PNG")
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)
        self.assertEqual(result["color_mode"], "RGB")

    def test_corrupted_image(self):
        image_path = self.test_dir / "corrupted.png"
        image_path.write_bytes(b"not a real image")

        result = analyze_image(image_path)

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_missing_image(self):
        image_path = self.test_dir / "missing.png"

        result = analyze_image(image_path)

        self.assertFalse(result["success"])
        self.assertIn("does not exist", result["error"])

    def test_unsupported_extension(self):
        image_path = self.test_dir / "document.txt"
        image_path.write_text("This is not an image.")

        result = analyze_image(image_path)

        self.assertFalse(result["success"])
        self.assertIn("Unsupported", result["error"])


if __name__ == "__main__":
    unittest.main()
