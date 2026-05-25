import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from backend.schemas import ClinicalSummary
load_dotenv()

class SummaryEngine:
    def __init__(self):
        self.llm = ChatGroq(
            api_key = os.getenv("GROQ_API_KEY"),
            model = "llama-3.3-70b-versatile",
            temperature = 0
        )
        self.structured_llm = self.llm.with_structured_output(ClinicalSummary)

    def get_analysis(self , text : str) -> ClinicalSummary:
        system_prompt = (
            "You are a clinical data analyst. Your job is to extract data from a medical report "
            "and organize it into a structured format. "
            "IMPORTANT: Compare the 'Result' with the 'Biological Reference Interval'. "
            "If the result is above the range, status is 'HIGH'. If below, 'LOW'. Otherwise 'NORMAL'. "
            "Explain medical terms in very simple language for a non-doctor."
        )
        return self.structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this report text:\n{text}"}
        ])


if __name__ == "__main__":
    from pdf_processor import extract_text_from_pdf

    # 1. Get text from your actual PDF
    raw_text = extract_text_from_pdf("backend/chaitanya_blood_report.pdf")

    # 2. Run the Engine
    engine = SummaryEngine()
    summary = engine.get_analysis(raw_text)

    # 3. Print the results
    print(f"Patient: {summary.patient_name}")
    print(f"Risk: {summary.risk_level}")
    print(f"Abnormal Results: {len(summary.abnormal_results)}")