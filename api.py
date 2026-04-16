from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import re

# Import your massive AI function
from local_qwen_reviewer import review_code

# Initialize the API server
app = FastAPI(title="Qwen Code Reviewer API")

# --- THE CORS FIX ---
# This allows your index.html file to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any website to connect (good for local testing)
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"],
)
# --------------------

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

class CodePatch(BaseModel):
    code: str
    language: str = "Java"


@app.post("/api/v1/review")
async def analyze_code(patch: CodePatch):
    try:
        print(f"Received {patch.language} code for review...")

        # 1. Get the raw XML-style string from the AI
        raw_ai_response = review_code(patch.code, patch.language)
        print(f"Raw AI Output: \n{raw_ai_response}\n")

        # 2. BULLETPROOF EXTRACTION (FORGIVING REGEX)
        # We make the closing tags optional and use lookaheads to stop at the next tag
        bug_found_match = re.search(r"<bug_found>(.*?)(?:</bug_found>|<severity>|$)", raw_ai_response,
                                    re.DOTALL | re.IGNORECASE)
        severity_match = re.search(r"<severity>(.*?)(?:</severity>|<issue>|$)", raw_ai_response,
                                   re.DOTALL | re.IGNORECASE)
        issue_match = re.search(r"<issue>(.*?)(?:</issue>|<fixed_code>|$)", raw_ai_response, re.DOTALL | re.IGNORECASE)
        fixed_code_match = re.search(r"<fixed_code>(.*?)(?:</fixed_code>|$)", raw_ai_response,
                                     re.DOTALL | re.IGNORECASE)

        # Fallback empty strings if the model completely missed a section
        bug_found_text = bug_found_match.group(1).strip() if bug_found_match else "false"
        severity_text = severity_match.group(1).strip() if severity_match else "UNKNOWN"
        issue_text = issue_match.group(1).strip() if issue_match else "AI failed to describe the issue."
        fixed_code_text = fixed_code_match.group(1).strip() if fixed_code_match else "AI failed to provide fixed code."

        # 3. Safely build the JSON object
        review_data = {
            "bug_found": "true" in bug_found_text.lower(),
            "severity": severity_text.upper(),
            "issue": issue_text,
            "fixed_code": fixed_code_text
        }

        # Clean up Markdown backticks if the AI stubbornly adds them
        if review_data["fixed_code"].startswith("```"):
            review_data["fixed_code"] = re.sub(r"^```[a-z]*\n|```$", "", review_data["fixed_code"],
                                               flags=re.MULTILINE).strip()

        return review_data

    except Exception as e:
        print(f"Error parsing AI response: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse AI output.")