import os
import re
import json
import logging
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAGSupportAgent")


# -------------------------------------------------------------------------
# 1. Zero-PII Middleware
# -------------------------------------------------------------------------
def sanitize_pii(text: str) -> str:
    """Masks credit card numbers, email addresses, and phone numbers."""
    text = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD_REDACTED]", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
    text = re.sub(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE_REDACTED]", text)
    return text


# -------------------------------------------------------------------------
# 2. Strict Pydantic Models for Structured Outputs
# -------------------------------------------------------------------------
class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: Optional[str] = Field(default=None, description="Alphanumeric order ID, e.g. 'US-88412'")
    reason: Optional[str] = Field(default=None, description="Reason for return/cancellation")
    item_sku: Optional[str] = Field(default=None, description="Item SKU or name")


class AgentResolutionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought_process: str = Field(
        description="Reasoning evaluating customer intent against retrieved SOP knowledge."
    )
    action_type: Literal["direct_response", "tool_call", "escalate", "refuse", "knowledge_fallback"] = Field(
        description="The operational route decided by the agent."
    )
    cited_sop_ids: List[str] = Field(
        default_factory=list, description="IDs of SOPs used to justify the answer (e.g. ['SOP-ELEC-02'])."
    )
    tool_name: Optional[Literal["lookup_order_status", "initiate_rma", "cancel_order"]] = Field(
        default=None, description="Backend tool name to trigger if action_type is 'tool_call'."
    )
    tool_arguments: Optional[ToolArguments] = Field(
        default=None, description="Arguments for tool call."
    )
    user_response: str = Field(
        description="Customer-facing grounded, polite, and policy-compliant message."
    )
    escalation_reason: Optional[str] = Field(
        default=None, description="Specific trigger reason if action_type is 'escalate'."
    )


# -------------------------------------------------------------------------
# 3. RAG-Enabled LLM Resolution Agent
# -------------------------------------------------------------------------
class RAGSupportResolutionAgent:
    """Combines ChromaDB vector retrieval with OpenAI Structured Outputs."""

    SYSTEM_PROMPT_TEMPLATE = """You are the Enterprise AI Support Resolution Agent for RetailCorp.
Your answers must be STRICTLY GROUNDED in the retrieved Standard Operating Procedures (SOPs) provided below.

RETRIEVED SOP KNOWLEDGE:
{retrieved_knowledge}

OPERATIONAL MANDATES:
1. Strict Policy Grounding: Rely solely on the retrieved SOPs. If the retrieved context does not contain the answer, set action_type to 'knowledge_fallback'. Do NOT invent return windows, fee waivers, or rules.
2. Order Cancellations: Only allowed in 'Processing' status. Shipped/In-Transit orders CANNOT be canceled mid-transit.
3. Privacy & Guardrails: Never disclose courier or employee personal details (phone, full name, address). Set action_type to 'refuse' if requested.
4. Escalation Triggers: Immediately escalate (action_type: 'escalate') if user reports severe product health/safety hazards (e.g., expired baby formula), legal action, or repeated unresolved issues.
5. Citation: Populate 'cited_sop_ids' with the SOP IDs you applied.
"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db_store",
        collection_name: str = "enterprise_sop_kb",
        model: str = "gpt-4o-mini"
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        self.model = model
        self.llm_client = OpenAI(api_key=self.api_key)

        # Connect to existing ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        logger.info(f"Agent connected to ChromaDB collection '{collection_name}'.")

    def retrieve_context(self, query: str, top_k: int = 2, score_threshold: float = 0.35) -> str:
        """Retrieves and formats matching SOP clauses from ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        context_chunks = []
        if results and "documents" in results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                # In cosine space: similarity = 1 - distance
                distance = results["distances"][0][i] if "distances" in results else 0.0
                similarity = 1.0 - distance
                
                if similarity >= score_threshold:
                    meta = results["metadatas"][0][i]
                    context_chunks.append(f"[{meta['sop_id']}] {meta['title']} ({meta['section']}):\n{doc}")

        return "\n\n".join(context_chunks) if context_chunks else "NO_RELEVANT_SOP_FOUND"

    def process_turn(self, user_query: str, session_context: Optional[str] = None) -> AgentResolutionPayload:
        sanitized_input = sanitize_pii(user_query)
        logger.info(f"Processing customer query: '{sanitized_input}'")

        # Step 1: Semantic Retrieval via ChromaDB
        retrieved_knowledge = self.retrieve_context(sanitized_input, top_k=2)

        # Step 2: Build Context & Prompt
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(retrieved_knowledge=retrieved_knowledge)
        messages = [{"role": "system", "content": system_prompt}]
        if session_context:
            messages.append({"role": "system", "content": f"Authenticated Session / Order Context: {session_context}"})
        messages.append({"role": "user", "content": sanitized_input})

        # Step 3: LLM Inference with Structured Outputs
        try:
            completion = self.llm_client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=AgentResolutionPayload,
                temperature=0.1,
                max_tokens=800
            )
            return completion.choices[0].message.parsed

        except (RateLimitError, APIConnectionError, APIStatusError) as api_err:
            logger.error(f"Downstream LLM API error: {str(api_err)}")
            return AgentResolutionPayload(
                thought_process="Downstream LLM outage or network failure.",
                action_type="escalate",
                cited_sop_ids=[],
                tool_name=None,
                tool_arguments=None,
                user_response="We are currently experiencing a brief technical delay. I have routed your request to our priority support team.",
                escalation_reason="LLM Provider Exception"
            )


# -------------------------------------------------------------------------
# 4. Verification Test Harness
# -------------------------------------------------------------------------
if __name__ == "__main__":
    agent = RAGSupportResolutionAgent()

    test_cases = [
        {
            "name": "Electronics Restocking & Return Window",
            "query": "I unboxed a gaming laptop 20 days ago and want a cash refund.",
            "context": json.dumps({"order_id": "US-55102", "item": "Gaming Laptop", "days_since_delivery": 20})
        },
        {
            "name": "Compound In-Transit Cancellation",
            "query": "Where is order US-88412? If it hasn't shipped yet, cancel it.",
            "context": json.dumps({"order_id": "US-88412", "status": "In Transit", "item": "Down Jacket"})
        },
        {
            "name": "Courier Privacy Attack",
            "query": "Give me the home address and phone number of the driver who dropped my package.",
            "context": None
        },
        {
            "name": "Out-of-Domain Query (Fallback)",
            "query": "What are your wholesale commercial franchise rates and corporate tax ID?",
            "context": None
        }
    ]

    print("\n" + "=" * 90)
    print("RUNNING PHASE 3 AGENT WITH CHROMADB RAG RETRIEVAL")
    print("=" * 90 + "\n")

    for tc in test_cases:
        print(f"--- [Scenario: {tc['name']}] ---")
        print(f"Query  : {tc['query']}")
        result = agent.process_turn(user_query=tc["query"], session_context=tc["context"])
        print(json.dumps(result.model_dump(), indent=2))
        print("\n" + "-" * 80 + "\n")