import unittest

from src.vision_backend import (
    _extract_json,
    _normalize_result,
    _validate_output,
    analyze_image_with_qwen,
)


class TestVisionBackend(unittest.TestCase):

    def test_extract_plain_json(self):
        text = '''
        {
            "image_type": "damage",
            "observations": ["螢幕有裂痕"],
            "visible_damage": "螢幕呈現蜘蛛網狀裂痕",
            "limitations": []
        }
        '''

        result = _extract_json(text)

        self.assertEqual(result["image_type"], "damage")
        self.assertEqual(result["visible_damage"], "螢幕呈現蜘蛛網狀裂痕")

    def test_extract_json_from_markdown(self):
        text = '''```json
        {
            "image_type": "product",
            "observations": ["筆記型電腦"],
            "visible_damage": null,
            "limitations": []
        }
        ```'''

        result = _extract_json(text)

        self.assertEqual(result["image_type"], "product")

    def test_extract_json_with_extra_text(self):
        text = '''
        以下是圖片分析結果：
        {
            "image_type": "receipt",
            "observations": ["收據"],
            "visible_damage": null,
            "limitations": []
        }
        分析完成。
        '''

        result = _extract_json(text)

        self.assertEqual(result["image_type"], "receipt")

    def test_extract_invalid_json(self):
        with self.assertRaises(ValueError):
            _extract_json("這裡沒有 JSON")

    def test_normalize_valid_result(self):
        raw = {
            "image_type": "damage",
            "observations": ["螢幕破裂"],
            "visible_damage": "螢幕有裂痕",
            "limitations": ["無法確認內部損壞"],
        }

        result = _normalize_result(raw)

        self.assertTrue(result["image_provided"])
        self.assertEqual(result["image_type"], "damage")
        self.assertEqual(result["visible_damage"], "螢幕有裂痕")

    def test_normalize_invalid_image_type(self):
        raw = {
            "image_type": "hacked_category",
            "observations": "not a list",
            "visible_damage": "",
            "limitations": None,
        }

        result = _normalize_result(raw)

        self.assertEqual(result["image_type"], "unknown")
        self.assertEqual(result["observations"], [])
        self.assertIsNone(result["visible_damage"])
        self.assertEqual(result["limitations"], [])

    def test_normalized_result_matches_schema(self):
        raw = {
            "image_type": "product",
            "observations": ["可見筆記型電腦"],
            "visible_damage": None,
            "limitations": [],
        }

        result = _normalize_result(raw)

        _validate_output(result)

    def test_corrupted_image_fails_safely(self):
        result = analyze_image_with_qwen("images/corrupted.png")

        self.assertTrue(result["image_provided"])
        self.assertEqual(result["image_type"], "unknown")
        self.assertEqual(result["observations"], [])
        self.assertTrue(result["limitations"])

    def test_missing_image_fails_safely(self):
        result = analyze_image_with_qwen("images/does_not_exist.png")

        self.assertTrue(result["image_provided"])
        self.assertEqual(result["image_type"], "unknown")
        self.assertTrue(result["limitations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
