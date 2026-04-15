from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import re

# Import your massive AI function
from local_qwen_reviewer import review_code

# Initialize the API server
app = FastAPI(title="Qwen Code Reviewer API")


class CodePatch(BaseModel):
    code: str
    language: str = "Java"


@app.post("/api/v1/review")
async def analyze_code(patch: CodePatch):
    try:
        print(f"Received {patch.language} code for review...")

        # 1. Get the raw string from the AI
        raw_ai_response = review_code(patch.code, patch.language)
        print(f"Raw AI Output: \n{raw_ai_response}\n")  # Let's print it to see what it's doing

        # 2. BULLETPROOF CLEANING: Find the first '{' and the last '}'
        start_idx = raw_ai_response.find("{")
        end_idx = raw_ai_response.rfind("}")

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in AI response")

        # Extract strictly the JSON block
        clean_json_string = raw_ai_response[start_idx:end_idx + 1]

        # 3. Parse the cleaned string
        review_data = json.loads(clean_json_string)

        return review_data

    except json.JSONDecodeError as e:
        print(f"JSON Parsing failed on this string: {clean_json_string}")
        raise HTTPException(status_code=500, detail="AI returned malformed JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))