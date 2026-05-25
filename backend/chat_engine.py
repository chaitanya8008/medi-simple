import os
from langchain_groq import ChatGroq
from backend.vector_store import MedicalVectorStore
from dotenv import load_dotenv

load_dotenv()


class ChatEngine:
    def __init__(self, vector_store: MedicalVectorStore):
        self.vector_store = vector_store
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0  # Slight temperature for conversational tone, but still very grounded
        )

        # Simple in-memory storage for chat history.
        # Keys will be session_ids (e.g., "chat_123"), Values will be lists of messages.
        self.chat_histories = {}

    def ask_question(self, session_id: str, question: str, global_summary: str) -> str:
        """
        Handles a user question with memory, retrieval, and guardrails.
        """
        # 1. Initialize memory for this session if it doesn't exist
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []

        history = self.chat_histories[session_id]

        # 2. Retrieve the most relevant chunks from our Vector Database
        # We only pull the top 5 to avoid overwhelming the LLM and saving tokens
        try:
            db = self.vector_store.load_existing()
            docs = db.similarity_search(question, k=5)
            # Extract just the enriched text from the retrieved documents
            retrieved_context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            retrieved_context = "No specific document context found. Please rely on the global summary."

        # 3. Format the chat history into a readable string for the prompt
        history_text = ""
        for msg in history[-4:]:  # Only keep the last 4 interactions to save tokens
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

        # 4. Construct the Super-Prompt
        system_prompt = f"""
                You are an AI Medical Assistant helping a user understand their clinical report.

                GLOBAL PATIENT SUMMARY:
                {global_summary}

                RELEVANT REPORT DATA (Retrieved based on the user's question):
                {retrieved_context}

                RECENT CONVERSATION HISTORY:
                {history_text}

                RULES:
                1. Answer the user's question simply and clearly, using the context provided.
                2. If the answer is not in the context, politely say you don't have that information. DO NOT guess.
                3. Explain medical terms in plain, reassuring English for a non-doctor.
                4. GUARDRAIL: You are an AI, not a doctor. If the user asks for a diagnosis, treatment, or asks "what should I do?", advise them to consult a doctor. 
                5. CRITICAL CONVERSATION RULE: Do NOT append a repetitive medical disclaimer to every message. Speak naturally. Only give a medical warning if absolutely necessary based on the user's current question. Do not sound like a robot.

                User's New Question: {question}
                """

        # 5. Get the answer from the LLM
        response = self.llm.invoke(system_prompt)
        answer = response.content.strip()

        # 6. Save this interaction to memory
        self.chat_histories[session_id].append({"role": "user", "content": question})
        self.chat_histories[session_id].append({"role": "assistant", "content": answer})

        return answer