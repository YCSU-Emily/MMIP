import sys
import unittest
from pathlib import Path
from jsonschema import validate

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import chatbot


class TestCustomerServiceBot(unittest.TestCase):

    def setUp(self):
        self.state = {
            "session_id": "test-session-001",
            "turn_count": 1,
            "previous_intent": None,
            "pending_intent": None
        }

    def test_order_tracking_missing_id(self):
        result, state = chatbot.build_output(
            "我要查詢訂單物流",
            self.state
        )

        self.assertEqual(
            result["intent"]["category"],
            "order_tracking"
        )

        self.assertTrue(
            result["intent"]["needs_clarification"]
        )

        self.assertEqual(
            result["conversation_state"]["missing_information"],
            ["order_id"]
        )

        self.assertEqual(
            result["response"]["message"],
            "請提供訂單編號，以便查詢物流。"
        )

        self.assertEqual(
            state["pending_intent"],
            "order_tracking"
        )

        validate(instance=result, schema=chatbot.SCHEMA)


    def test_order_tracking_multi_turn(self):
        # Turn 1: User asks about order tracking.
        result1, state = chatbot.build_output(
            "我的 ASUS 筆電無法開機，我想查詢訂單物流",
            self.state
        )

        self.assertEqual(
            result1["intent"]["category"],
            "order_tracking"
        )

        self.assertEqual(
            state["pending_intent"],
            "order_tracking"
        )

        # Turn 2: User provides the missing order ID.
        state["turn_count"] += 1

        result2, state = chatbot.build_output(
            "AB123456",
            state
        )

        self.assertEqual(
            result2["intent"]["category"],
            "order_tracking"
        )

        self.assertEqual(
            result2["entities"]["order_id"],
            "AB123456"
        )

        self.assertEqual(
            result2["entities"]["problem_description"],
            "我的 ASUS 筆電無法開機，我想查詢訂單物流"
        )

        self.assertFalse(
            result2["intent"]["needs_clarification"]
        )

        self.assertIsNone(
            state["pending_intent"]
        )

        self.assertEqual(
            result2["routing"]["action"],
            "track_order"
        )

        validate(instance=result1, schema=chatbot.SCHEMA)
        validate(instance=result2, schema=chatbot.SCHEMA)


    def test_repair_request_multi_turn_preserves_description(self):
        # Turn 1: User describes a product problem without a brand.
        result1, state = chatbot.build_output(
            "我的產品有問題",
            self.state.copy()
        )

        self.assertEqual(
            result1["intent"]["category"],
            "repair_request"
        )
        self.assertTrue(
            result1["intent"]["needs_clarification"]
        )
        self.assertEqual(
            result1["entities"]["problem_description"],
            "我的產品有問題"
        )
        self.assertEqual(
            state["pending_intent"],
            "repair_request"
        )

        # Turn 2: User provides the product brand.
        state["turn_count"] += 1

        result2, state = chatbot.build_output(
            "ASUS",
            state
        )

        self.assertEqual(
            result2["intent"]["category"],
            "repair_request"
        )
        self.assertEqual(
            result2["entities"]["product_name"],
            "ASUS"
        )
        self.assertEqual(
            result2["entities"]["problem_description"],
            "我的產品有問題"
        )
        self.assertFalse(
            result2["intent"]["needs_clarification"]
        )
        self.assertEqual(
            result2["routing"]["department"],
            "repair_center"
        )
        self.assertEqual(
            result2["routing"]["action"],
            "create_repair_request"
        )
        self.assertIsNone(state["pending_intent"])

        validate(instance=result1, schema=chatbot.SCHEMA)
        validate(instance=result2, schema=chatbot.SCHEMA)


    def test_repair_request(self):
        result, state = chatbot.build_output(
            "我的 ASUS 筆電壞掉了",
            self.state
        )

        self.assertEqual(
            result["intent"]["category"],
            "repair_request"
        )

        self.assertEqual(
            result["entities"]["product_name"],
            "ASUS"
        )

        self.assertEqual(
            result["routing"]["department"],
            "repair_center"
        )

        self.assertEqual(
            result["routing"]["action"],
            "create_repair_request"
        )

        self.assertFalse(
            result["intent"]["needs_clarification"]
        )

        validate(instance=result, schema=chatbot.SCHEMA)


    def test_unknown_intent(self):
        result, state = chatbot.build_output(
            "我有一個問題",
            self.state
        )

        self.assertEqual(
            result["intent"]["category"],
            "unknown"
        )

        self.assertTrue(
            result["intent"]["needs_clarification"]
        )

        self.assertEqual(
            result["routing"]["action"],
            "ask_follow_up"
        )

        self.assertTrue(
            result["conversation_state"]["awaiting_user_input"]
        )

        self.assertEqual(
            state["pending_intent"],
            "unknown"
        )

        validate(instance=result, schema=chatbot.SCHEMA)


    def test_session_reset(self):
        # Create a pending order-tracking task.
        result, state = chatbot.build_output(
            "我要查詢訂單物流",
            self.state
        )

        self.assertEqual(
            state["pending_intent"],
            "order_tracking"
        )

        # Simulate the state reset performed by /reset.
        new_state = {
            "session_id": "new-test-session",
            "turn_count": 0,
            "previous_intent": None,
            "pending_intent": None
        }

        self.assertNotEqual(
            new_state["session_id"],
            state["session_id"]
        )

        self.assertEqual(new_state["turn_count"], 0)
        self.assertIsNone(new_state["previous_intent"])
        self.assertIsNone(new_state["pending_intent"])
        self.assertIsNone(new_state.get("order_id"))


    def test_warranty_check(self):
        result, state = chatbot.build_output(
            "我想查詢 ASUS 產品的保固期限",
            self.state
        )

        self.assertEqual(
            result["intent"]["category"],
            "warranty_check"
        )
        self.assertFalse(
            result["intent"]["needs_clarification"]
        )
        self.assertEqual(
            result["routing"]["department"],
            "warranty_service"
        )
        self.assertEqual(
            result["routing"]["action"],
            "check_warranty"
        )
        self.assertFalse(result["routing"]["requires_human"])
        self.assertIsNone(state["pending_intent"])

        validate(instance=result, schema=chatbot.SCHEMA)


    def test_return_refund(self):
        result, state = chatbot.build_output(
            "我要申請退貨退款",
            self.state
        )

        self.assertEqual(
            result["intent"]["category"],
            "return_refund"
        )
        self.assertFalse(
            result["intent"]["needs_clarification"]
        )
        self.assertEqual(
            result["routing"]["department"],
            "returns_refunds"
        )
        self.assertEqual(
            result["routing"]["action"],
            "start_return_refund"
        )
        self.assertFalse(result["routing"]["requires_human"])
        self.assertIsNone(state["pending_intent"])

        validate(instance=result, schema=chatbot.SCHEMA)


    def test_human_handoff(self):
        result, state = chatbot.build_output(
            "我想找真人客服協助",
            self.state
        )

        self.assertEqual(
            result["intent"]["category"],
            "human_handoff"
        )
        self.assertFalse(
            result["intent"]["needs_clarification"]
        )
        self.assertEqual(
            result["routing"]["department"],
            "human_agent"
        )
        self.assertEqual(
            result["routing"]["action"],
            "handoff_to_human"
        )
        self.assertTrue(result["routing"]["requires_human"])
        self.assertIsNone(state["pending_intent"])

        validate(instance=result, schema=chatbot.SCHEMA)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMacBookEntityRecognition(unittest.TestCase):

    def test_macbook_repair_request_recognizes_product(self):
        state = {
            "session_id": "macbook-test",
            "turn_count": 1,
            "previous_intent": None,
            "pending_intent": None
        }

        result, updated_state = chatbot.build_output(
            "我的 MacBook Pro 螢幕摔裂了，想申請維修。",
            state
        )

        self.assertEqual(updated_state["product_name"], "Apple MacBook")
        self.assertEqual(result["intent"]["category"], "repair_request")
        self.assertFalse(result["intent"]["needs_clarification"])
        self.assertEqual(
            result["routing"]["department"],
            "repair_center"
        )
