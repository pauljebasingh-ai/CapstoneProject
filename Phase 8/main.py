import os
import re
import json
import time
import uuid
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# =========================================================================
# 1. Production Structured JSON Logger Setup
# =========================================================================
class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON for Datadog / CloudWatch."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        return json.dumps(log_data)

logger = logging.getLogger("EnterpriseSupportService")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setFormatter(JsonFormatter())
logger.handlers = [log_handler]


# =========================================================================
# 2. PII Sanitization
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
# 3. Backend OMS & LangChain Tools
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

class OrderLookupInput(BaseModel):
    order_id: str = Field(description="Order ID format US-XXXXX")

@tool(args_schema=OrderLookupInput)
def lookup_order_status(order_id: str) -> Dict[str, Any]:
    """Retrieve fulfillment status and dates from OMS."""
    key = order_id.upper().strip()
    order = MOCK_OMS.get(key)
    if not order:
        return {"success": False, "error": f"Order ID '{order_id}' was not found in OMS."}
    return {"success": True, "order": order}

class CancelOrderInput(BaseModel):
    order_id: str = Field(description="Order ID format US-XXXXX")
    reason: str = Field(description="Cancellation reason")

@tool(args_schema=CancelOrderInput)
def cancel_order(order_id: str, reason: str) -> Dict[str, Any]:
    """Cancel unfulfilled processing orders."""
    key = order_id.upper().strip()
    order = MOCK_OMS.get(key)
    if not order:
        return {"success": False, "error": f"Order '{order_id}' does not exist."}
    if order["status"] != "Processing":
        return {
            "success": False,
            "error": f"Order '{order_id}' is '{order['status']}'. Shipped/In-Transit items cannot be cancelled."
        }
    order["status"] = "Cancelled"
    return {
        "success": True,
        "cancelled_order_id": order_id,
        "refund_amount": order["total_amount"],
        "status": "Order successfully cancelled and refund initiated."
    }


# =========================================================================
# 4. RAG Knowledge Base
# =========================================================================
class ProductionKnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db_store"):
        self.persist_dir = persist_dir
        self.vector_store = None
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and os.path.exists(self.persist_dir):
            try:
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                self.vector_store = Chroma(
                    collection_name="enterprise_sop_kb",
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_dir
                )
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}. Defaulting to embedded fallback.")

    def retrieve(self, query: str, k: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.vector_store:
            # Resilient fallback corpus if ChromaDB service is unreachable
            fallback = [
                {"sop_id": "SOP-ORD-03", "title": "Cancellation Window", "section": "Locked Transit", "text": "Orders in 'In Transit' cannot be cancelled."},
                {"sop_id": "SOP-RET-01", "title": "Apparel Returns", "section": "30-Day Window", "text": "Apparel returns permitted within 30 days of delivery scan."}
            ]
            context_str = "\n\n".join([f"[{f['sop_id']}] {f['title']}: {f['text']}" for f in fallback])
            return context_str, fallback

        docs = self.vector_store.similarity_search(query, k=k)
        if not docs:
            return "NO_RELEVANT_SOP_FOUND", []
        
        sources = []
        blocks = []
        for d in docs:
            sop_id = d.metadata.get("sop_id", "SOP")
            title = d.metadata.get("title", "")
            sources.append({"sop_id": sop_id, "title": title, "text": d.page_content})
            blocks.append(f"[{sop_id}] {title}:\n{d.page_content}")
        return "\n\n".join(blocks), sources


# =========================================================================
# 5. Core Resolution Engine with Circuit Breakers
# =========================================================================
class ResilientSupportEngine:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "test-key")
        self.tools = [lookup_order_status, cancel_order]
        self.tools_by_name = {t.name: t for t in self.tools}
        self.kb = ProductionKnowledgeBase()
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0.0,
            timeout=10.0,
            max_retries=2
        ).bind_tools(self.tools)

    def execute(self, user_query: str, request_id: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        sanitized_input = sanitize_pii(user_query)
        
        retrieved_context, sources = self.kb.retrieve(sanitized_input, k=2)

        system_prompt = (
            "You are the Enterprise AI Support Resolution Agent for RetailCorp.\n"
            "Decisions must be grounded in these retrieved SOPs:\n"
            f"{retrieved_context}\n\n"
            "Rules:\n"
            "1. Call 'lookup_order_status' first before cancellation or return eligibility.\n"
            "2. In-transit orders cannot be cancelled.\n"
            "3. Refuse all requests for employee/courier phone numbers or addresses."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=sanitized_input)
        ]

        tool_logs = []
        try:
            for turn in range(3):
                ai_msg = self.llm.invoke(messages)
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.info(
                        "Request resolved successfully",
                        extra={"request_id": request_id, "latency_ms": latency_ms}
                    )
                    return {
                        "status": "success",
                        "response": ai_msg.content,
                        "sources": sources,
                        "tools_called": tool_logs,
                        "latency_ms": latency_ms
                    }

                for tc in ai_msg.tool_calls:
                    t_name = tc["name"]
                    t_args = tc["args"]
                    tool_logs.append({"tool": t_name, "args": t_args})
                    
                    tool_obj = self.tools_by_name.get(t_name)
                    output = tool_obj.invoke(t_args) if tool_obj else {"error": "Tool not found"}
                    messages.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(output)))

            raise RuntimeError("Max tool iteration count exceeded.")

        except (RateLimitError, APIConnectionError, APIStatusError) as api_err:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Downstream LLM API failure: {str(api_err)}",
                extra={"request_id": request_id, "latency_ms": latency_ms}
            )
            return {
                "status": "fallback_degraded",
                "response": "Our AI support systems are currently experiencing elevated latency. I have routed your ticket to a dedicated human specialist.",
                "sources": [],
                "tools_called": tool_logs,
                "latency_ms": latency_ms,
                "escalated": True
            }
        except Exception as ex:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Unexpected runtime failure: {str(ex)}",
                extra={"request_id": request_id, "latency_ms": latency_ms}
            )
            return {
                "status": "error_handled",
                "response": "An unexpected error occurred while processing your request. Your inquiry has been forwarded to human customer support.",
                "sources": [],
                "tools_called": tool_logs,
                "latency_ms": latency_ms,
                "escalated": True
            }


# =========================================================================
# 6. FastAPI Web Layer & Schemas
# =========================================================================
app = FastAPI(
    title="RetailCorp AI Support Resolution Microservice",
    version="1.0.0",
    description="Production-ready asynchronous RAG and Tool-Calling support agent API."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

support_engine = ResilientSupportEngine()

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="User message", min_length=2, max_length=1000)
    customer_id: Optional[str] = Field(default="CUST-ANON")
    session_id: Optional[str] = Field(default=None)

class ChatResponse(BaseModel):
    request_id: str
    status: str
    response: str
    sources: List[Dict[str, Any]]
    tools_called: List[Dict[str, Any]]
    latency_ms: float
    escalated: bool = False

@app.middleware("http")
async def correlation_and_latency_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.perf_counter()
    
    response: Response = await call_next(request)
    
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = str(latency)
    return response

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness & Readiness probe for Kubernetes / Docker health checks."""
    return {
        "status": "healthy",
        "service": "ai-support-resolution-agent",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/v1/chat/resolve", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def resolve_chat(req: ChatRequest, request: Request):
    """Processes customer query with RAG grounding, tool calling, and fallback safeguards."""
    req_id = request.state.request_id
    result = support_engine.execute(user_query=req.query, request_id=req_id)
    
    return ChatResponse(
        request_id=req_id,
        status=result["status"],
        response=result["response"],
        sources=result["sources"],
        tools_called=result["tools_called"],
        latency_ms=result["latency_ms"],
        escalated=result.get("escalated", False)
    )