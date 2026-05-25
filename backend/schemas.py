from pydantic import BaseModel, Field
from typing import List, Optional

class LabResult(BaseModel):
    test_name: str = Field(description="Name of the test (e.g., RBC Count, Lymphocytes)")
    value: str = Field(description="The actual numeric or qualitative result")
    unit: str = Field(description="Unit of measurement (e.g., %, gm/dL)")
    reference_range: str = Field(description="The normal range (e.g., 12.0 - 16.0)")
    status: str = Field(description="Status: 'HIGH', 'LOW', or 'NORMAL' based on the range")

class ClinicalSummary(BaseModel):
    patient_name: Optional[str] = None
    report_date: Optional[str] = None 
    key_findings: List[str] = Field(description="3-5 critical medical insights")
    abnormal_results: List[LabResult] = Field(description="List of tests that are HIGH or LOW")
    normal_results: List[LabResult] = Field(description="List of tests within normal range")
    risk_level: str = Field(description="Overall clinical risk: 'LOW', 'MODERATE', or 'HIGH'")
    summary_text: str = Field(description="A 2-sentence simple explanation for the patient")