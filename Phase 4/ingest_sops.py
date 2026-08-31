import os
import re
import logging
from typing import List
from docx import Document
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SOPIngestion")


class SOPChunk(BaseModel):
    chunk_id: str
    sop_id: str
    title: str
    category: str
    version: str
    section_name: str
    content: str
    searchable_text: str


class DocxSOPParser:
    """Parses RetailCorp SOP .docx documents and extracts chunked clauses."""

    @staticmethod
    def parse_docx(file_path: str) -> List[SOPChunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

        doc = Document(file_path)
        chunks: List[SOPChunk] = []

        current_sop_id = "UNKNOWN"
        current_title = ""
        current_category = "General"
        current_version = "1.0"
        
        current_section = "Overview"
        section_paragraphs: List[str] = []
        chunk_counter = 1

        def flush_current_section():
            nonlocal chunk_counter
            if section_paragraphs:
                body_text = "\n".join(section_paragraphs).strip()
                if body_text:
                    c_id = f"{current_sop_id}_c{chunk_counter}"
                    searchable = (
                        f"SOP: [{current_sop_id}] {current_title} | "
                        f"Category: {current_category} | "
                        f"Section: {current_section}\n{body_text}"
                    )
                    chunks.append(
                        SOPChunk(
                            chunk_id=c_id,
                            sop_id=current_sop_id,
                            title=current_title,
                            category=current_category,
                            version=current_version,
                            section_name=current_section,
                            content=body_text,
                            searchable_text=searchable
                        )
                    )
                    chunk_counter += 1
                section_paragraphs.clear()

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            # Detect Heading: [SOP-ID] Title
            if p.style.name.startswith("Heading 1") or re.match(r"^\[SOP-[A-Z]+-\d+\]", text):
                flush_current_section()
                chunk_counter = 1
                match = re.match(r"^\[(SOP-[A-Z]+-\d+)\]\s*(.*)", text)
                if match:
                    current_sop_id = match.group(1)
                    current_title = match.group(2)
                else:
                    current_sop_id = f"SOP-{chunk_counter}"
                    current_title = text
                current_section = "Overview"

            # Detect Metadata
            elif "Category:" in text and "SOP ID:" in text:
                cat_match = re.search(r"Category:\s*([^|]+)", text)
                ver_match = re.search(r"Version:\s*([^|]+)", text)
                if cat_match:
                    current_category = cat_match.group(1).strip()
                if ver_match:
                    current_version = ver_match.group(1).strip()

            # Detect Bullet Clauses
            elif text.startswith("•") or text.startswith("-"):
                flush_current_section()
                bullet_clean = text.lstrip("•- ").strip()
                if ":" in bullet_clean:
                    sec_title, sec_body = bullet_clean.split(":", 1)
                    current_section = sec_title.strip()
                    section_paragraphs.append(sec_body.strip())
                else:
                    current_section = "Policy Rule"
                    section_paragraphs.append(bullet_clean)
            else:
                section_paragraphs.append(text)

        flush_current_section()
        logger.info(f"Extracted {len(chunks)} chunks from '{file_path}'.")
        return chunks


def run_ingestion(
    docx_file: str = "./Phase 4/RetailCorp_Standard_Operating_Procedures_Master.docx",
    persist_dir: str = "./chroma_db_store",
    collection_name: str = "enterprise_sop_kb"
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")

    logger.info(f"Connecting to ChromaDB at: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    chunks = DocxSOPParser.parse_docx(docx_file)
    if not chunks:
        logger.warning("No chunks found to ingest.")
        return

    logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB collection '{collection_name}'...")
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.searchable_text for c in chunks],
        metadatas=[
            {
                "sop_id": c.sop_id,
                "title": c.title,
                "category": c.category,
                "version": c.version,
                "section": c.section_name
            }
            for c in chunks
        ]
    )
    logger.info(f"Ingestion successful! Total documents in vector store: {collection.count()}")


if __name__ == "__main__":
    run_ingestion()