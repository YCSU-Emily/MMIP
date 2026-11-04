import unittest
from unittest.mock import patch

from jsonschema import validate
from src.chatbot import build_output, SCHEMA


def make_state():
    return {
        "session_id": "qwen-integration-test",
        "turn_count": 1,
        "previous_intent": None,
        "pending_intent": None
    }


class TestChatbotQwenIntegration(unittest.TestCase):

    def test_valid_image_calls_qwen_and_integrates_result(self):
        mock_result = {
            "image_provided": True,
            "image_type": "damage",
            "observations": [
                "A MacBook Pro laptop is visible.",
                "The screen has extensive spiderweb cracks."
            ],
            "visible_damage": "Severe screen fracture",
            "limitations": [
                "Visual evidence does not confirm warranty eligibility."
            ]
        }

        with patch(
            "src.chatbot.analyze_image_with_qwen",
            return_value=mock_result
        ) as mock_qwen:

            result, _ = build_output(
                "我的 MacBook Pro 螢幕摔裂了，想申請維修。",
                make_state(),
                image_path="images/real_product.jpg",
                enable_vision=True
            )

        mock_qwen.assert_called_once_with(
            "images/real_product.jpg",
            user_text="我的 MacBook Pro 螢幕摔裂了，想申請維修。"
        )

        validate(instance=result, schema=SCHEMA)

        self.assertEqual(
            result["image_analysis"]["image_type"],
            "damage"
        )
        self.assertEqual(
            result["image_analysis"]["visible_damage"],
            "Severe screen fracture"
        )
        self.assertEqual(
            len(result["image_analysis"]["observations"]),
            2
        )

    def test_corrupted_image_does_not_call_qwen(self):
        with patch(
            "src.chatbot.analyze_image_with_qwen"
        ) as mock_qwen:

            result, _ = build_output(
                "我的產品壞掉了",
                make_state(),
                image_path="images/corrupted.png",
                enable_vision=True
            )

        mock_qwen.assert_not_called()

        validate(instance=result, schema=SCHEMA)

        self.assertEqual(
            result["image_analysis"]["image_type"],
            "unknown"
        )
        self.assertTrue(
            any(
                "驗證失敗" in item
                for item in result["image_analysis"]["limitations"]
            )
        )

    def test_text_only_does_not_call_qwen(self):
        with patch(
            "src.chatbot.analyze_image_with_qwen"
        ) as mock_qwen:

            result, _ = build_output(
                "我要查詢訂單物流",
                make_state(),
                enable_vision=True
            )

        mock_qwen.assert_not_called()

        validate(instance=result, schema=SCHEMA)

        self.assertFalse(
            result["image_analysis"]["image_provided"]
        )
        self.assertEqual(
            result["image_analysis"]["image_type"],
            "none"
        )


if __name__ == "__main__":
    unittest.main()
