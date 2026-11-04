import unittest

from jsonschema import validate
from src.chatbot import build_output, SCHEMA


def make_state():
    return {
        "session_id": "image-test",
        "turn_count": 1,
        "previous_intent": None,
        "pending_intent": None
    }


class TestChatbotImageIntegration(unittest.TestCase):

    def test_text_only(self):
        result, _ = build_output(
            "我要查詢訂單物流",
            make_state()
        )

        validate(instance=result, schema=SCHEMA)

        image = result["image_analysis"]
        self.assertFalse(image["image_provided"])
        self.assertEqual(image["image_type"], "none")

    def test_valid_image(self):
        result, _ = build_output(
            "我的產品有問題",
            make_state(),
            "images/sample_product.png"
        )

        validate(instance=result, schema=SCHEMA)

        image = result["image_analysis"]
        self.assertTrue(image["image_provided"])
        self.assertEqual(image["image_type"], "unknown")
        self.assertTrue(any("PNG" in item for item in image["observations"]))
        self.assertTrue(any("640x480" in item for item in image["observations"]))

    def test_damage_image_classification(self):
        result, _ = build_output(
            "我的筆電螢幕裂了",
            make_state(),
            "images/sample_product.png"
        )
        self.assertEqual(result["image_analysis"]["image_type"], "damage")
        validate(instance=result, schema=SCHEMA)

    def test_receipt_image_classification(self):
        result, _ = build_output(
            "這是我的購買收據",
            make_state(),
            "images/sample_product.png"
        )
        self.assertEqual(result["image_analysis"]["image_type"], "receipt")
        validate(instance=result, schema=SCHEMA)

    def test_order_image_classification(self):
        result, _ = build_output(
            "這是我的訂單截圖",
            make_state(),
            "images/sample_product.png"
        )
        self.assertEqual(
            result["image_analysis"]["image_type"],
            "order_information"
        )
        validate(instance=result, schema=SCHEMA)

    def test_product_image_classification(self):
        result, _ = build_output(
            "這是我的產品照片",
            make_state(),
            "images/sample_product.png"
        )
        self.assertEqual(result["image_analysis"]["image_type"], "product")
        validate(instance=result, schema=SCHEMA)

    def test_corrupted_image(self):
        result, _ = build_output(
            "我的產品有問題",
            make_state(),
            "images/corrupted.png"
        )

        validate(instance=result, schema=SCHEMA)

        image = result["image_analysis"]
        self.assertTrue(image["image_provided"])
        self.assertTrue(
            any("驗證失敗" in item for item in image["limitations"])
        )


if __name__ == "__main__":
    unittest.main()
