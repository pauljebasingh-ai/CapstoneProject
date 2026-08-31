import os
import re
import json
import logging
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Phase7-AdaptiveAgent")


# =========================================================================
# 1. PII Sanitization
# =========================================================================
def sanitize_pii(text: str) -> str:
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD_REDACTED]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
    text = re.sub(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "[PHONE_REDACTED]",
        text
    )
    return text


# =========================================================================
# 2. Mock Backend OMS
# =========================================================================
MOCK_OMS = {
    "US-55102": {
        "order_id": "US-55102",
        "customer_id": "CUST-901",
        "item_name": "Gaming Laptop Pro 15",
        "status": "Delivered",
        "shipped_date": "2026-08-08",
        "delivery_scan_date": "2026-08-11",
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
# 3. LangChain Tools
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


# =========================================================================
# 4. Adaptive Feedback Store & Preference Schema
# =========================================================================
class CustomerFeedbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    customer_id: str
    csat_score: int = Field(ge=1, le=5, description="1 to 5 star rating")
    feedback_text: str = Field(description="User qualitative comment or complaint")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AdaptivePersonaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verbosity: Literal["concise_bullet_points", "balanced_standard", "detailed_explanatory"] = "balanced_standard"
    tone_preference: Literal["direct_efficiency", "standard_polite", "high_empathy_supportive"] = "standard_polite"
    proactive_suggestions: bool = True
    technical_depth: Literal["low_layman", "medium", "high_technical"] = "medium"
    learned_preferences_summary: List[str] = Field(default_factory=list)


class FeedbackAdaptationEngine:
    """Stores customer feedback and derives behavioral adaptations."""

    def __init__(self, llm_client: ChatOpenAI):
        self.llm = llm_client
        self.feedback_database: List[Dict[str, Any]] = []
        self.profiles: Dict[str, AdaptivePersonaProfile] = {
            "CUST-901": AdaptivePersonaProfile()
        }

    def record_feedback(self, feedback: CustomerFeedbackPayload):
        """Persists feedback and runs adaptation analysis."""
        self.feedback_database.append(feedback.model_dump())
        logger.info(f"Feedback recorded for {feedback.customer_id}: CSAT={feedback.csat_score} | '{feedback.feedback_text}'")
        self._adapt_profile(feedback.customer_id, feedback)

    def _adapt_profile(self, customer_id: str, latest_feedback: CustomerFeedbackPayload):
        """Analyzes feedback to update customer behavioral profile."""
        current_profile = self.profiles.get(customer_id, AdaptivePersonaProfile())

        analysis_prompt = (
            f"You are an AI Behavioral Profiler. Analyze this customer support feedback and update their persona settings.\n\n"
            f"FEEDBACK TEXT: \"{latest_feedback.feedback_text}\"\n"
            f"CSAT SCORE: {latest_feedback.csat_score}/5\n"
            f"CURRENT SETTINGS: {current_profile.model_dump_json()}\n\n"
            f"Update their settings to resolve friction. If the user complains about long explanations or fluff, change verbosity to 'concise_bullet_points' and tone to 'direct_efficiency'."
        )

        structured_llm = self.llm.with_structured_output(AdaptivePersonaProfile)
        updated_profile = structured_llm.invoke(analysis_prompt)

        self.profiles[customer_id] = updated_profile
        logger.info(f"Updated Adaptive Persona for {customer_id}: {updated_profile.model_dump_json()}")

    def get_profile(self, customer_id: str) -> AdaptivePersonaProfile:
        return self.profiles.get(customer_id, AdaptivePersonaProfile())


# =========================================================================
# 5. Chroma RAG Knowledge Base
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
        #         "[SOP-RET-01] Apparel: 30-day return window. Flat $5.99 return label fee.\n"
        #         "[SOP-ELEC-02] Electronics: 15-day return window. 10% restocking fee if opened. Refer to 1-Yr OEM warranty after 15 days.\n"
        #         "[SOP-ORD-03] Cancellations: Allowed only in 'Processing'. Locked in transit.\n"
        #         "[SOP-LOG-04] Lost Shipments: Requires 7 consecutive business days without tracking scan before refund."
        #     )
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([f"[{d.metadata.get('sop_id', 'SOP')}] {d.page_content}" for d in docs]) if docs else "NO_RELEVANT_SOP_FOUND"


# =========================================================================
# 6. Adaptive Support Resolution Agent
# =========================================================================
class AdaptiveSupportAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm_base = ChatOpenAI(model=model_name, temperature=0.0)
        self.tools = [lookup_order_status]
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm_base.bind_tools(self.tools)
        self.kb = LangChainKnowledgeBase()
        self.adaptation_engine = FeedbackAdaptationEngine(llm_client=self.llm_base)

    def process_turn(self, user_query: str, customer_id: str = "CUST-901") -> Dict[str, Any]:
        sanitized_input = sanitize_pii(user_query)
        retrieved_sops = self.kb.retrieve_context(sanitized_input, k=2)
        persona = self.adaptation_engine.get_profile(customer_id)

        # Dynamic Persona Prompt Directives
        persona_instructions = (
            f"DYNAMIC ADAPTIVE BEHAVIORAL DIRECTIVES:\n"
            f"• Verbosity Level: {persona.verbosity.upper()}\n"
            f"• Tone Mode: {persona.tone_preference.upper()}\n"
            f"• Proactive Next-Steps: {'Enabled' if persona.proactive_suggestions else 'Disabled'}\n"
            f"• Special Instructions: {'; '.join(persona.learned_preferences_summary) if persona.learned_preferences_summary else 'Follow standard support protocol.'}\n"
        )

        system_prompt = (
            "You are the Enterprise AI Support Resolution Agent for RetailCorp.\n"
            "Your decisions must be strictly grounded in the retrieved Standard Operating Procedures below.\n\n"
            f"{persona_instructions}\n\n"
            "RETRIEVED SOP KNOWLEDGE:\n"
            f"{retrieved_sops}\n\n"
            "OPERATIONAL RULES:\n"
            "1. Tool Usage: Always call 'lookup_order_status' first before determining return/cancellation eligibility.\n"
            "2. Grounding: Rely strictly on the SOP text. Do not invent return windows or refund policies.\n"
            "3. Persona Adaptation: Strictly follow the Tone Mode and Verbosity Level indicated above."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=sanitized_input)
        ]

        tool_logs = []
        for _ in range(3):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                final_response = ai_msg.content
                break

            for tc in ai_msg.tool_calls:
                t_name = tc["name"]
                t_args = tc["args"]
                tool_obj = self.tools_by_name.get(t_name)
                tool_output = tool_obj.invoke(t_args) if tool_obj else {"error": "Tool not found"}
                tool_logs.append({"tool": t_name, "args": t_args, "result": tool_output})
                messages.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(tool_output)))
        else:
            final_response = "Request exceeded step threshold."

        return {
            "query": sanitized_input,
            "persona_applied": persona.model_dump(),
            "tools_executed": tool_logs,
            "agent_response": final_response
        }


# =========================================================================
# 7. Before vs. After Demonstration Harness
# =========================================================================
if __name__ == "__main__":
    agent = AdaptiveSupportAgent()
    customer_id = "CUST-901"
    test_query = "What is the status of my order US-55102 and can I get a refund for it?"

    print("\n" + "=" * 95)
    print("PHASE 7: ADAPTIVE BEHAVIOUR & FEEDBACK TUNING DEMONSTRATION")
    print("=" * 95 + "\n")

    # -------------------------------------------------------------
    # 1. BEFORE ADAPTATION (Default Baseline Persona)
    # -------------------------------------------------------------
    print("===========================================================================================")
    print("1. BEFORE ADAPTATION (Default Polite / Balanced Persona)")
    print("===========================================================================================")
    res_before = agent.process_turn(user_query=test_query, customer_id=customer_id)
    print(f"Applied Persona : {res_before['persona_applied']['tone_preference']} | {res_before['persona_applied']['verbosity']}")
    print(f"\n[AGENT RESPONSE (BEFORE)]:\n{res_before['agent_response']}\n")

    # -------------------------------------------------------------
    # 2. FEEDBACK INGESTION (Negative CSAT + Friction Signal)
    # -------------------------------------------------------------
    print("===========================================================================================")
    print("2. INGESTING CUSTOMER FEEDBACK & UPDATING PERSONA PROFILE")
    print("===========================================================================================")
    feedback = CustomerFeedbackPayload(
        session_id="SESS-001",
        customer_id=customer_id,
        csat_score=2,
        feedback_text="Too much robotic pleasantries and long-winded paragraphs. Just give me the direct facts, bullet points, and no fluffy apologies."
    )
    agent.adaptation_engine.record_feedback(feedback)
    print("Feedback successfully parsed and stored in long-term customer profile.\n")

    # -------------------------------------------------------------
    # 3. AFTER ADAPTATION (Adapted Persona: Direct & Concise)
    # -------------------------------------------------------------
    print("===========================================================================================")
    print("3. AFTER ADAPTATION (Optimized Persona: Direct Efficiency & Bullet Points)")
    print("===========================================================================================")
    res_after = agent.process_turn(user_query=test_query, customer_id=customer_id)
    print(f"Applied Persona : {res_after['persona_applied']['tone_preference']} | {res_after['persona_applied']['verbosity']}")
    print(f"\n[AGENT RESPONSE (AFTER)]:\n{res_after['agent_response']}\n")