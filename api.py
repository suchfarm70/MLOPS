from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# Import your massive AI function from the other file
# (Make sure your other file is named local_qwen_reviewer.py)
from local_qwen_reviewer import review_code

# Initialize the API server
app = FastAPI(title="Qwen Code Reviewer API")


# Define the data structure we expect from the Spring Boot backend
class CodePatch(BaseModel):
    code: str
    language: str = "Java"


@app.post("/api/v1/review")
async def analyze_code(patch: CodePatch):
    try:
        print(f"Received {patch.language} code for review...")

        # Pass the code to your local RTX 3050 brain
        raw_ai_response = review_code(patch.code, patch.language)

        # Parse the strict JSON string back into a Python dictionary
        review_data = json.loads(raw_ai_response)

        return review_data

    except json.JSONDecodeError:
        # Just in case the AI hallucinates and breaks the JSON format
        raise HTTPException(status_code=500, detail="AI failed to return valid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this, we will use uvicorn in the terminal!