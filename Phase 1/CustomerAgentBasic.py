import json
import logging
import re
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

class BaselineSupportAgent:
    """
    Phase 2 Basic Rule-Based Customer Support AI Agent.
    Relies entirely on static keywords, rigid templates, and regex routing.
    """

    def __init__(self):
        # Static Knowledge Base / Policy Dictionary
        self.knowledge_base = {
            "password_reset": "Verify user identity via registered email, then instruct user to navigate to /auth/reset.",
            "refund_policy": "Full refunds are permitted within 14 days of purchase for unused digital licenses.",
            "payment_failure": "Check gateway error codes. Recommend customer retry with an alternate credit card.",
        }

    def process_query(self, user_query: str) -> dict:
        query_clean = user_query.strip()
        matched_intent = None
        response_text = ""
        is_escalated = False

        # Rigid keyword routing rules
        if re.search(r"\b(password|login|reset)\b", query_clean, re.IGNORECASE):
            matched_intent = "password_issue"
            response_text = f"Troubleshooting Steps: {self.knowledge_base['password_reset']}"
        elif re.search(r"\b(refund|money back)\b", query_clean, re.IGNORECASE):
            matched_intent = "refund_inquiry"
            response_text = f"Policy Reference: {self.knowledge_base['refund_policy']}"
        elif re.search(r"\b(payment|failed|transaction)\b", query_clean, re.IGNORECASE):
            matched_intent = "payment_issue"
            response_text = f"Troubleshooting Steps: {self.knowledge_base['payment_failure']}"
        else:
            matched_intent = "unknown"
            is_escalated = True
            response_text = "Standard Fallback: Case routed to general human queue due to unmapped intent."

        interaction_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_query": query_clean,
            "detected_intent": matched_intent,
            "generated_response": response_text,
            "escalated": is_escalated,
            "architecture_mode": "regex_rule_baseline",
        }

        # Log interaction for audit tracking
        logging.info("Agent Execution Log: %s", json.dumps(interaction_payload))
        return interaction_payload


if __name__ == "__main__":
    agent = BaselineSupportAgent()

    # 4 Test scenarios matching the Phase 1 test cases
    test_cases = [
        "The customer cannot login even after resetting their password.",
        "The customer is asking for a refund on a subscription.",
        "The customer says they believe their account has been compromised. Can I reset everything for them?",
        "Customer wants compensation for an undocumented delay.",
    ]

    print("\n=== RUNNING PHASE 2 BASELINE AGENT TEST HARNESS ===\n")
    for test in test_cases:
        print(f"User Query: {test}")
        result = agent.process_query(test)
        print(f"Agent Response: {result['generated_response']}")
        print(f"Escalated: {result['escalated']}\n" + "-" * 60)