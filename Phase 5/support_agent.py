import os
import re
import json
import logging
from typing import Optional, Dict, Any, List, Tuple

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LangChainRAGAgent")


# -------------------------------------------------------------------------
# 1. Zero-PII Sanitization Middleware
# -------------------------------------------------------------------------
def sanitize_pii(text: str) -> str:
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD_REDACTED]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
    text = re.sub(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "[PHONE_REDACTED]",
        text
    )
    return text


# -------------------------------------------------------------------------
# 2. Mock Backend OMS Database
# -------------------------------------------------------------------------
MOCK_OMS = {
    "US-88412": {
        "order_id": "US-88412",
        "item_name": "Winter Down Jacket",
        "status": "In Transit",
        "shipped_date": "2026-08-28",
        "delivery_scan_date": None,
        "total_amount": 149.99
    },
    "US-55102": {
        "order_id": "US-55102",
        "item_name": "Gaming Laptop Pro 15",
        "status": "Delivered",
        "shipped_date": "2026-08-08",
        "delivery_scan_date": "2026-08-11",
        "total_amount": 1299.00
    },
    "US-10293": {
        "order_id": "US-10293",
        "item_name": "Running Shoes",
        "status": "Processing",
        "shipped_date": None,
        "delivery_scan_date": None,
        "total_amount": 89.50
    }
}


# -------------------------------------------------------------------------
# 3. LangChain Typed Tools
# -------------------------------------------------------------------------
class OrderLookupInput(BaseModel):
    order_id: str = Field(description="The alphanumeric order ID, e.g., 'US-88412'")

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
        description="Must be set to True only if customer explicitly approved return generation."
    )

@tool(args_schema=RMAInput)
def initiate_rma(order_id: str, reason: str, confirmed_by_customer: bool = False) -> Dict[str, Any]:
    """Generate an authorized RMA return label for delivered orders."""
    if not confirmed_by_customer:
        return {
            "success": False,
            "error": "Confirmation required. Ask the user if they want you to generate the RMA label before creating it."
        }
    key = order_id.upper().strip()
    order = MOCK_OMS.get(key)
    if not order:
        return {"success": False, "error": f"Order '{order_id}' does not exist."}
    if order["status"] != "Delivered":
        return {
            "success": False,
            "error": f"Order status is '{order['status']}'. Returns are only permitted for 'Delivered' items."
        }
    rma_code = f"RMA-{key[3:]}-998"
    return {
        "success": True,
        "rma_code": rma_code,
        "label_url": f"https://shipping.retailcorp.com/labels/{rma_code}.pdf"
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
            "error": f"Order '{order_id}' is '{order['status']}'. Shipped/In-Transit orders cannot be canceled."
        }
    order["status"] = "Cancelled"
    return {
        "success": True,
        "cancelled_order_id": order_id,
        "refund_amount": order["total_amount"],
        "status": "Order successfully cancelled and refund initiated."
    }


# -------------------------------------------------------------------------
# 4. LangChain Chroma RAG Knowledge Base with Source Metadata
# -------------------------------------------------------------------------
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

    def retrieve_context_with_sources(self, query: str, k: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        """Returns both formatted prompt string and structured source documents."""
        # if not self.vector_store:
        #     # Fallback mock corpus
        #     fallback_sources = [
        #         {
        #             "sop_id": "SOP-ORD-03",
        #             "title": "Order Cancellation and Modification Window",
        #             "section": "Locked Transit Stage",
        #             "text": "Once an order transitions to 'In Transit' or 'Fulfilled', warehouse automation locks the shipment. In-transit orders cannot be canceled."
        #         },
        #         {
        #             "sop_id": "SOP-ELEC-02",
        #             "title": "Consumer Electronics Return and Warranty",
        #             "section": "15-Day Return Window & Restocking Fee",
        #             "text": "Consumer electronics have a strict 15-day return window. 10% restocking fee applies to unsealed items. Refer to 1-Yr OEM warranty past 15 days."
        #         }
        #     ]
        #     context_str = "\n\n".join([f"[{s['sop_id']}] {s['title']} ({s['section']}):\n{s['text']}" for s in fallback_sources])
        #     return context_str, fallback_sources

        docs = self.vector_store.similarity_search(query, k=k)
        if not docs:
            return "NO_RELEVANT_SOP_FOUND", []

        sources = []
        context_blocks = []
        for d in docs:
            sop_id = d.metadata.get("sop_id", "SOP-GEN")
            title = d.metadata.get("title", "Standard Operating Procedure")
            section = d.metadata.get("section", "General")
            content = d.page_content

            sources.append({
                "sop_id": sop_id,
                "title": title,
                "section": section,
                "text": content
            })
            context_blocks.append(f"[{sop_id}] {title} ({section}):\n{content}")

        return "\n\n".join(context_blocks), sources


# -------------------------------------------------------------------------
# 5. Agent Engine with Visible Source of Truth Display
# -------------------------------------------------------------------------
class GroundedSupportResolutionAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.tools = [lookup_order_status, initiate_rma, cancel_order]
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = ChatOpenAI(model=model_name, temperature=0.0).bind_tools(self.tools)
        self.kb = LangChainKnowledgeBase()

    def process_turn(self, user_query: str) -> Dict[str, Any]:
        sanitized_input = sanitize_pii(user_query)
        
        # 1. Retrieve RAG context and source documents
        retrieved_context, source_docs = self.kb.retrieve_context_with_sources(sanitized_input, k=2)

        # 2. Build system instructions with grounding bounds
        system_prompt = (
            "You are the Enterprise AI Support Resolution Agent for RetailCorp.\n"
            "Your decisions must be strictly grounded in the retrieved Standard Operating Procedures below.\n\n"
            "RETRIEVED SOP KNOWLEDGE:\n"
            f"{retrieved_context}\n\n"
            "OPERATIONAL RULES:\n"
            "1. Tool Usage: Always call 'lookup_order_status' first before determining return/cancellation eligibility.\n"
            "2. Grounding: Rely strictly on the SOP text. Do not invent return windows or refund policies.\n"
            "3. Cancellations: In-transit or delivered orders CANNOT be canceled.\n"
            "4. Privacy Guardrails: Never disclose courier phone numbers, full names, or addresses. Refuse immediately.\n"
            "5. Escalation: Escalate immediately if customer reports safety hazards or threatens legal action."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=sanitized_input)
        ]

        tool_execution_log = []

        # 3. Tool Calling Loop (Max 4 iterations)
        for _ in range(4):
            ai_msg = self.llm.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                final_response = ai_msg.content
                break

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_obj = self.tools_by_name.get(tool_name)

                tool_execution_log.append({"tool": tool_name, "args": tool_args})

                if tool_obj:
                    tool_output = tool_obj.invoke(tool_args)
                else:
                    tool_output = f"Error: Tool '{tool_name}' not recognized."

                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=json.dumps(tool_output) if isinstance(tool_output, dict) else str(tool_output)
                    )
                )
        else:
            final_response = "I could not complete your request within operational limits. Escalating to human support."

        return {
            "query": sanitized_input,
            "agent_response": final_response,
            "ground_truth_sources": source_docs,
            "tool_calls_executed": tool_execution_log
        }


# -------------------------------------------------------------------------
# 6. Pretty Print Formatter
# -------------------------------------------------------------------------
def display_agent_turn(result: Dict[str, Any]):
    print("=" * 90)
    print(f"USER QUERY: {result['query']}")
    print("=" * 90)
    
    print("\n🔍 [RAG GROUND TRUTH / SOURCE OF TRUTH APPLIED]:")
    if result["ground_truth_sources"]:
        for idx, src in enumerate(result["ground_truth_sources"], 1):
            print(f"  [{idx}] SOP ID     : {src['sop_id']} - {src['title']}")
            print(f"      Section    : {src['section']}")
            print(f"      Policy Text: {src['text'].replace(chr(10), ' ')}")
    else:
        print("  None (No matching SOP found in vector database; fallback rule applied)")

    if result["tool_calls_executed"]:
        print("\n⚙️  [TOOLS EXECUTED]:")
        for tc in result["tool_calls_executed"]:
            print(f"  • Tool: {tc['tool']} | Args: {tc['args']}")

    print("\n💬 [FINAL AGENT RESPONSE]:")
    print(result["agent_response"])
    print("\n" + "-" * 90 + "\n")


# -------------------------------------------------------------------------
# 7. Verification Test Suite
# -------------------------------------------------------------------------
if __name__ == "__main__":
    agent = GroundedSupportResolutionAgent()

    test_queries = [
        "Check where US-88412 is, and if it has not shipped yet, cancel it immediately.",
        "I bought a gaming laptop (US-55102) 20 days ago and opened it. Can I get a full refund?",
        #"Please cancel my order US-10293 right now, I ordered by mistake.",
        "I placed order US-10293 for running shoes earlier today, but I changed my mind and want to return them for my money back."
    ]

    for q in test_queries:
        turn_result = agent.process_turn(q)
        display_agent_turn(turn_result)