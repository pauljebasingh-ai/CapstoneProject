import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

# Configure baseline execution logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BaselineSupportAgent")


class BaselineSupportAgent:
    """
    Rule-based customer support agent using deterministic regex matching 
    and static templates. Represents a legacy Tier-1 chatbot baseline.
    """

    def __init__(self, mock_db: Optional[Dict[str, Any]] = None):
        # Mock backend order database
        self.mock_db = mock_db or {
            "US-88412": {"item": "Winter Jacket", "status": "In Transit", "days_since_order": 3},
            "US-10293": {"item": "Running Shoes", "status": "Delivered", "days_since_order": 8},
            "US-77120": {"item": "Leather Boots", "status": "Delivered", "days_since_order": 5},
            "US-55102": {"item": "Gaming Laptop", "status": "Delivered", "days_since_order": 92}
        }
        
        # Static Policy Mapping
        self.RETURN_WINDOW_DAYS = 30

    def parse_order_id(self, user_text: str) -> Optional[str]:
        """Extract order number using simple regex pattern."""
        match = re.search(r"US-\d{5}", user_text, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def respond(self, user_input: str) -> str:
        """
        Evaluate user input against static rule hierarchy and return templated string.
        """
        # Baseline logging flaw: Logs raw input directly (PII risk)
        logger.info(f"Incoming request received: '{user_input}'")

        text = user_input.lower()
        order_id = self.parse_order_id(user_input)

        # Rule 1: Order Tracking
        if "track" in text or "where is" in text or "status" in text:
            if order_id:
                order = self.mock_db.get(order_id)
                if order:
                    response = (
                        f"Order {order_id} ({order['item']}) is currently: {order['status']}. "
                        f"It was placed {order['days_since_order']} days ago."
                    )
                else:
                    response = f"I could not locate Order {order_id} in our database. Please check the number."
            else:
                response = "Please provide your 7-character Order ID (e.g., US-12345) so I can look up the tracking status."

        # Rule 2: Returns and Refunds
        elif "return" in text or "refund" in text:
            if order_id:
                order = self.mock_db.get(order_id)
                if order:
                    if order["days_since_order"] <= self.RETURN_WINDOW_DAYS:
                        response = (
                            f"Order {order_id} is eligible for return. I have generated return label RMA-{order_id[3:]}-RET. "
                            f"Please drop it off at any authorized carrier point."
                        )
                    else:
                        response = (
                            f"Order {order_id} was delivered {order['days_since_order']} days ago. "
                            f"Our return policy strictly allows returns within {self.RETURN_WINDOW_DAYS} days."
                        )
                else:
                    response = f"I cannot process a return without a valid matching Order ID for {order_id}."
            else:
                response = "To start a return or refund, please state your Order ID (e.g., US-12345)."

        # Rule 3: Direct Agent Escalation Request
        elif "manager" in text or "human" in text or "agent" in text or "supervisor" in text:
            response = "I am transferring your request to our Tier-2 human support queue. An agent will respond shortly."

        # Rule 4: Fallback
        else:
            response = (
                "I am an automated assistant. I can help with order tracking and return requests. "
                "Please specify what you need help with along with your Order ID."
            )

        # Baseline logging flaw: Raw output logged without compliance audit checks
        logger.info(f"Agent response dispatched: '{response}'")
        return response


# --- Sample Test Execution Suite ---
if __name__ == "__main__":
    agent = BaselineSupportAgent()

    test_conversations = [
        # Test 1: Successful Single-Intent Tracking
        ("Standard Lookup", "Hi, can you track my package US-88412?"),
        
        # Test 2: Ambiguity Failure (Missing Order ID, no session history)
        ("Ambiguity Flaw", "I want to return the shoes I bought last week, they don't fit."),
        
        # Test 3: Multi-Intent / Reasoning Failure
        ("Compound Intent Flaw", "Check where US-88412 is, and if it has not shipped yet, please cancel it immediately."),
        
        # Test 4: Privacy / PII Ingestion Flaw
        ("PII Logging Flaw", "My email is john.doe@example.com and phone is +1-555-0199. Cancel my order US-88412."),
        
        # Test 5: Policy Boundary Verification
        ("Strict Policy Check", "Can I get a refund for order US-55102? It was delivered 92 days ago.")
    ]

    print("\n" + "="*80)
    print("BASELINE AGENT EXECUTION LOGS")
    print("="*80 + "\n")

    for category, prompt in test_conversations:
        print(f"--- [Scenario: {category}] ---")
        print(f"User  : {prompt}")
        output = agent.respond(prompt)
        print(f"Agent : {output}\n")