from pydantic import BaseModel

class InputData(BaseModel):
    age: int
    region: str
    maritalStatus: str
    householdSize: int
    employmentStatus: str
    educationLevel: str
    loanAmount: float
    repaidAmount: float
    collectionEfforts: str
    originalInterestRate: float
    currentInterestRate: float
    repaymentPlan: str
    
