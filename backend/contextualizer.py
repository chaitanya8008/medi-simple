from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class MedicalChunker:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[". ", " ", ""]
        )

    def split_text(self, text: str) -> list[str]:
        """
        Splits medical text smartly, prioritizing line breaks to keep lab rows intact.
        """
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_length = 0

        for line in lines:
            line_len = len(line) + 1

            if line_len > self.chunk_size:
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
                    current_length = 0

                fallback_chunks = self.fallback_splitter.split_text(line)
                chunks.extend(fallback_chunks)
                continue

            if current_length + line_len > self.chunk_size and current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))

                while current_length > self.chunk_overlap and len(current_chunk_lines) > 1:
                    removed_line = current_chunk_lines.pop(0)
                    current_length -= (len(removed_line) + 1)

            current_chunk_lines.append(line)
            current_length += line_len

        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

        return [chunk.strip() for chunk in chunks if chunk.strip()]


class Contextualizer:
    def __init__(self):
        # We use the smart 70b model for accurate medical context generation
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    def _generate_chunk_context(self, document_summary: str, chunk_text: str) -> str:
        """
        Uses the short global summary + the specific chunk to generate a 1-2 sentence context.
        """
        prompt = f"""
        You are a medical data assistant preparing data for a search engine.

        Global Patient Summary:
        {document_summary}

        Specific Document Chunk:
        {chunk_text}

        Task: Write a 1-2 sentence context explaining what medical data this specific chunk contains and who it belongs to. 
        Do NOT start with "This chunk contains...". Just state the facts.
        Example: "Complete Blood Count (CBC) results for Mr. Chaitanya, specifically detailing White Blood Cell differentials."
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating context: {e}")
            return "Medical report data snippet."

    def process_document(self, raw_text: str, document_summary: str) -> list[dict]:
        """
        Runs the smart chunker and enriches each chunk with LLM-generated context.
        """
        # 1. Initialize our smart medical chunker
        chunker = MedicalChunker(chunk_size=800, chunk_overlap=150)
        raw_chunks = chunker.split_text(raw_text)

        print(f"Total smart chunks to enrich: {len(raw_chunks)}")
        enriched_chunks = []

        # 2. Enrich each chunk
        for i, chunk in enumerate(raw_chunks):
            print(f"Enriching chunk {i + 1}/{len(raw_chunks)}...")

            # Generate the context using ONLY the short summary and the chunk
            context = self._generate_chunk_context(document_summary, chunk)

            # 3. Build the structured output object
            chunk_data = {
                "content": chunk,
                "context": context,
                "enriched": f"{context}\n\n{chunk}",
                "metadata": {
                    "chunk_id": i,
                    "source": "medical_report",
                    "type": "lab_result" if "Result" in chunk or "Interval" in chunk else "text"
                }
            }
            enriched_chunks.append(chunk_data)

            # Small delay to keep Groq Free Tier happy
            time.sleep(1)

        return enriched_chunks
