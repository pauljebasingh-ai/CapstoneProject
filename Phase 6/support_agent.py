import os
import re
import json
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Phase6-PlanningMemoryAgent")


# =========================================================================
# 1. Zero-PII Middleware
# =========================================================================
def sanitize_pii(text: str) -> str:
    """Masks payment card numbers, emails, and phone numbers."""
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD_REDACTED]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
    text = re.sub(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "[PHONE_REDACTED]",
        text
    )
    return text


# =========================================================================
# 2. Mock OMS Backend Database
# =========================================================================
MOCK_OMS = {
    "US-88412": {
        "order_id": "US-88412",
        "customer_id": "CUST-901",
        "item_name": "Winter Down Jacket",
        "status": "In Transit",
        "shipped_date": "2026-08-28",
        "delivery_scan_date": None,
        "total_amount": 149.99
    },
    "US-55102": {
        "order_id": "US-55102",
        "customer_id": "CUST-901",
        "item_name": "Gaming Laptop Pro 15",
        "status": "Delivered",
        "shipped_date": "2026-08-08",
        "delivery_scan_date": "2026-08-11",  # 20 days ago
        "total_amount": 1299.00
    },
    "US-10293": {
        "order_id": "US-10293",
        "customer_id": "CUST-901",
        "item_name": "Running Shoes",
        "status": "Processing",
        "shipped_date": None,
        "delivery_scan_date": None,
        "total_amount": 89.50
    }
}


# =========================================================================
# 3. LangChain Typed Tools with Guardrails
# =========================================================================
class OrderLookupInput(BaseModel):
    order_id: str = Field(description="Order ID matching format US-XXXXX")

@tool(args_schema=OrderLookupInput)
def lookup_order_status(order_id: str) -> Dict[str, Any]:
    """Lookup real-time order state, fulfillment details, and delivery timestamps."""
    key = order_id.upper().strip()
    order = MOCK_OMS.get(key)
    if not order:
        return {"success": False, "error": f"Order ID '{order_id}' was not found in OMS."}
    return {"success": True, "order": order}


class RMAInput(BaseModel):
    order_id: str = Field(description="Order ID, e.g., 'US-55102'")
    reason: str = Field(description="Customer return reason")
    confirmed_by_customer: bool = Field(
        default=False,
        description="Must be set to True only if customer explicitly confirmed return generation."
    )

@tool(args_schema=RMAInput)
def initiate_rma(order_id: str, reason: str, confirmed_by_customer: bool = False) -> Dict[str, Any]:
    """Generate an authorized RMA return label for delivered orders."""
    if not confirmed_by_customer:
        return {
            "success": False,
            "requires_confirmation": True,
            "error": "Confirmation required. Ask the user if they want you to generate the RMA label before creating it."
        }

    key = order_id.upper().strip()
    order = MOCK_OMS.get(key)
    if not order:
        return {"success": False, "error": f"Order '{order_id}' does not exist."}
    
    if order["status"] != "Delivered":
        return {
            "success": False,
            "error": f"Order status is '{order['status']}'. Returns are only permitted for 'Delivered' items[cite: 3]."
        }

    rma_code = f"RMA-{key[3:]}-998"
    return {
        "success": True,
        "rma_code": rma_code,
        "label_url": f"https://shipping.retailcorp.com/labels/{rma_code}.pdf",
        "instructions": "Package item with original tags and drop off at carrier depot within 14 days[cite: 3]."
    }


class CancelOrderInput(BaseModel):
    order_id: str = Field(description="Order ID to cancel, e.g., 'US-10293'")
    reason: str = Field(description="Reason for cancellation")

@tool(args_schema=CancelOrderInput)
def cancel_order(order_id: str, reason: str) -> Dict[str, Any]:
    """Cancel an unfulfilled order currently in 'Processing' status."""
    key = order_id.upper().strip()
    order = MOCK_OMS.get(key)
    if not order:
        return {"success": False, "error": f"Order '{order_id}' does not exist."}
    
    if order["status"] != "Processing":
        return {
            "success": False,
            "error": f"Order '{order_id}' is in status '{order['status']}'. Shipped/In-Transit orders cannot be canceled[cite: 3]."
        }

    order["status"] = "Cancelled"
    return {
        "success": True,
        "cancelled_order_id": order_id,
        "refund_amount": order["total_amount"],
        "status": "Order successfully cancelled and full refund initiated."
    }


# =========================================================================
# 4. Short-Term & Long-Term Memory Architecture with TTL & Reset Rules
# =========================================================================
class SessionContext(BaseModel):
    session_id: str
    customer_id: str
    active_order_id: Optional[str] = None
    active_item_name: Optional[str] = None
    pending_action: Optional[str] = None
    last_interaction_timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryManager:
    """Manages short-term conversation buffers, entity tracking, and persistent profiles."""

    def __init__(self, max_history_turns: int = 5, session_ttl_minutes: int = 30):
        self.max_history_turns = max_history_turns
        self.session_ttl_minutes = session_ttl_minutes
        
        # Short-term memory: session_id -> List[BaseMessage]
        self.short_term_store: Dict[str, List[Any]] = {}
        
        # Structured Session State: session_id -> SessionContext
        self.session_state_store: Dict[str, SessionContext] = {}

        # Long-term memory: customer_id -> Profile dictionary
        self.long_term_profile_store: Dict[str, Dict[str, Any]] = {
            "CUST-901": {
                "name": "Alex",
                "tier": "Gold VIP Member",
                "preferred_channel": "Email / Chat",
                "past_escalations_count": 0,
                "notes": "Prefers fast resolution; verified payment method on file."
            }
        }

    def get_or_create_session(self, session_id: str, customer_id: str = "CUST-901") -> SessionContext:
        """Retrieves or initializes session, checking TTL expiry."""
        now = datetime.utcnow()
        if session_id in self.session_state_store:
            session = self.session_state_store[session_id]
            # Check TTL
            if now - session.last_interaction_timestamp > timedelta(minutes=self.session_ttl_minutes):
                logger.info(f"[MEMORY RETENTION] Session {session_id} expired (> {self.session_ttl_minutes} mins). Resetting.")
                self.reset_session(session_id)
                session = SessionContext(session_id=session_id, customer_id=customer_id)
                self.session_state_store[session_id] = session
            else:
                session.last_interaction_timestamp = now
        else:
            session = SessionContext(session_id=session_id, customer_id=customer_id)
            self.session_state_store[session_id] = session

        return session

    def add_turn(self, session_id: str, human_msg: str, ai_msg: str):
        """Appends turn to sliding-window buffer."""
        if session_id not in self.short_term_store:
            self.short_term_store[session_id] = []

        history = self.short_term_store[session_id]
        history.append(HumanMessage(content=human_msg))
        history.append(AIMessage(content=ai_msg))

        # Enforce sliding window (keep last N turns = 2*N messages)
        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            self.short_term_store[session_id] = history[-max_messages:]
            logger.info(f"[MEMORY SLIDING WINDOW] Trimmed history for {session_id} to {max_messages} messages.")

    def get_history(self, session_id: str) -> List[Any]:
        return self.short_term_store.get(session_id, [])

    def reset_session(self, session_id: str):
        """Explicit memory clear trigger."""
        if session_id in self.short_term_store:
            del self.short_term_store[session_id]
        if session_id in self.session_state_store:
            del self.session_state_store[session_id]
        logger.info(f"[MEMORY RESET] Session {session_id} memory cleared.")

    def update_session_entities(self, session_id: str, order_id: Optional[str] = None, item: Optional[str] = None, pending_action: Optional[str] = None):
        """Tracks conversational entities for anaphoric resolution."""
        if session_id in self.session_state_store:
            s = self.session_state_store[session_id]
            if order_id:
                s.active_order_id = order_id
            if item:
                s.active_item_name = item
            if pending_action is not None:
                s.pending_action = pending_action

    def get_long_term_profile(self, customer_id: str) -> Dict[str, Any]:
        return self.long_term_profile_store.get(customer_id, {})


# =========================================================================
# 5. Chroma RAG Store
# =========================================================================
class LangChainKnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db_store"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                collection_name="enterprise_sop_kb",
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vector_store = None

    def retrieve_context(self, query: str, k: int = 2) -> str:
        # if not self.vector_store:
        #     return (
        #         "[SOP-RET-01] Apparel Return Policy: 30-day return window from delivery scan[cite: 3]. $5.99 label deduction unless defective[cite: 3].\n"
        #         "[SOP-ELEC-02] Electronics Policy: 15-day return window[cite: 3]. 10% restocking fee if unsealed[cite: 3]. Refer to 1-Yr OEM warranty past 15 days[cite: 3].\n"
        #         "[SOP-ORD-03] Order Cancellations: Allowed ONLY in 'Processing' status[cite: 3]. Mid-transit cancellations locked[cite: 3].\n"
        #         "[SOP-LOG-04] Lost Shipments: Requires 7 consecutive business days without scan update[cite: 3].\n"
        #         "[SOP-SAFE-05] Data Privacy: Never disclose courier phone number, full name, or address[cite: 3]."
        #     )

        docs = self.vector_store.similarity_search(query, k=k)
        if not docs:
            return "NO_RELEVANT_SOP_FOUND"
        return "\n\n".join([f"[{d.metadata.get('sop_id', 'SOP')}] {d.page_content}" for d in docs])


# =========================================================================
# 6. Planning, Memory & Multi-Step Reasoning Agent
# =========================================================================
class PlanningMemorySupportAgent:
    """Combines Task Decomposition, Short/Long-Term Memory, and Tool Execution."""

    PLANNER_SYSTEM_PROMPT = """You are the Senior Task Planner for RetailCorp AI Support.
Analyze the user's multi-turn request, conversational history, and active entity context.
Deconstruct the request into an explicit sequential plan before executing tools.

Your structured thought output MUST follow this exact format:
[GOAL]: High-level customer objective.
[CONTEXT ENTITIES]: Active Order ID, Item, and State.
[EXECUTION PLAN]:
  Step 1: Check/resolve referenced order.
  Step 2: Retrieve relevant SOP policy.
  Step 3: Execute tool or request confirmation.
  Step 4: Formulate customer-facing answer.
"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.tools = [lookup_order_status, initiate_rma, cancel_order]
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = ChatOpenAI(model=model_name, temperature=0.0).bind_tools(self.tools)
        self.kb = LangChainKnowledgeBase()
        self.memory = MemoryManager(max_history_turns=4, session_ttl_minutes=30)

    def process_turn(self, session_id: str, user_query: str, customer_id: str = "CUST-901") -> Dict[str, Any]:
        sanitized_input = sanitize_pii(user_query)

        # 1. Retrieve Memory & Session Context
        session = self.memory.get_or_create_session(session_id=session_id, customer_id=customer_id)
        chat_history = self.memory.get_history(session_id=session_id)
        profile = self.memory.get_long_term_profile(customer_id=customer_id)

        # Extract order mentions or resolve anaphora
        found_orders = re.findall(r"US-\d{5}", sanitized_input.upper())
        if found_orders:
            session.active_order_id = found_orders[0]

        # 2. Retrieve Grounded SOP Context
        retrieved_sops = self.kb.retrieve_context(sanitized_input, k=2)

        # 3. Assemble Planning & Memory Context Block
        system_instructions = (
            "You are the Enterprise AI Support Resolution Agent for RetailCorp.\n"
            "You possess explicit multi-step reasoning, conversation memory, and transactional tool access.\n\n"
            "CUSTOMER PROFILE (LONG-TERM MEMORY):\n"
            f"• Customer: {profile.get('name', 'Valued Customer')} ({profile.get('tier', 'Standard')})\n"
            f"• Notes: {profile.get('notes', 'None')}\n\n"
            "ACTIVE SESSION ENTITIES (SHORT-TERM MEMORY):\n"
            f"• Active Order ID in Focus: {session.active_order_id or 'None'}\n"
            f"• Active Item: {session.active_item_name or 'None'}\n"
            f"• Pending Action: {session.pending_action or 'None'}\n\n"
            "RETRIEVED SOP KNOWLEDGE BASE:\n"
            f"{retrieved_sops}\n\n"
            "OPERATIONAL RULES & PLANNING PROTOCOL:\n"
            "1. Multi-Step Reasoning: Always check order state and match with SOP rules before executing mutations[cite: 3].\n"
            "2. Anaphoric References: If the user says 'cancel it' or 'return that', apply the action to the Active Order ID in Focus.\n"
            "3. Confirmation Gating: If user says 'Yes, proceed', and there is a pending RMA or cancellation action, proceed with tool execution.\n"
            "4. Grounding: Never invent refund or return parameters outside the retrieved SOPs[cite: 3].\n"
            "5. Privacy: Never disclose courier phone numbers or personal addresses[cite: 3]."
        )

        messages: List[Any] = [SystemMessage(content=system_instructions)]
        # Inject conversational history
        messages.extend(chat_history)
        # Inject current turn
        messages.append(HumanMessage(content=sanitized_input))

        tool_execution_log = []

        # 4. Multi-Step Execution Loop
        for iteration in range(4):
            ai_msg = self.llm.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                final_response = ai_msg.content
                break

            for tool_call in ai_msg.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]

                # Auto-fill active_order_id if omitted in anaphoric tool calls
                if "order_id" in t_args and not t_args["order_id"] and session.active_order_id:
                    t_args["order_id"] = session.active_order_id

                tool_execution_log.append({"tool": t_name, "args": t_args})
                logger.info(f"[Turn Iteration {iteration+1}] Calling Tool: {t_name}({t_args})")

                tool_obj = self.tools_by_name.get(t_name)
                if tool_obj:
                    tool_output = tool_obj.invoke(t_args)
                else:
                    tool_output = {"success": False, "error": f"Tool '{t_name}' not recognized."}

                # Update session entity state based on tool outputs
                if t_name == "lookup_order_status" and isinstance(tool_output, dict) and tool_output.get("success"):
                    order_data = tool_output.get("order", {})
                    self.memory.update_session_entities(
                        session_id=session_id,
                        order_id=order_data.get("order_id"),
                        item=order_data.get("item_name")
                    )

                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=json.dumps(tool_output) if isinstance(tool_output, dict) else str(tool_output)
                    )
                )
        else:
            final_response = "I apologize, but this multi-step request exceeded operational limits. Connecting you to human support."

        # 5. Persist dialogue to Short-Term Memory Buffer
        self.memory.add_turn(session_id=session_id, human_msg=sanitized_input, ai_msg=final_response)

        return {
            "session_id": session_id,
            "query": sanitized_input,
            "active_entity_state": {
                "active_order_id": session.active_order_id,
                "active_item_name": session.active_item_name,
                "pending_action": session.pending_action
            },
            "tools_executed": tool_execution_log,
            "agent_response": final_response
        }


# =========================================================================
# 7. Multi-Turn Verification Harness
# =========================================================================
def run_multi_turn_demo():
    agent = PlanningMemorySupportAgent()
    session_id = "SESSION_TEST_101"

    conversation_turns = [
        # Turn 1: Lookup and establish order in conversational context
        "Hi, can you check the status of my order US-10293?",

        # Turn 2: Anaphoric reference ("it", "the shoes") + policy exploration
        "I realized I ordered the wrong size. Can you cancel it for me?",

        # Turn 3: Context retention and confirmation check
        "Thank you. Can you confirm what was refunded and if there is anything else on that order?"
    ]

    print("\n" + "=" * 95)
    print("PHASE 6: MULTI-TURN CONVERSATION & PLANNING EVALUATION")
    print("=" * 95 + "\n")

    for idx, user_input in enumerate(conversation_turns, 1):
        print(f"-------------------------------------------------------------------------------------------")
        print(f"[TURN {idx}] CUSTOMER: {user_input}")
        print(f"-------------------------------------------------------------------------------------------")

        result = agent.process_turn(session_id=session_id, user_query=user_input)

        print(f"🧠 [SESSION ENTITY STATE]: {result['active_entity_state']}")
        if result["tools_executed"]:
            print(f"⚙️  [TOOLS EXECUTED]     : {result['tools_executed']}")
        print(f"\n🤖 [AGENT RESPONSE]:\n{result['agent_response']}\n")


if __name__ == "__main__":
    run_multi_turn_demo()