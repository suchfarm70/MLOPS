import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. DOWNLOAD & LOAD THE QWEN BRAIN (GPU OPTIMIZED)
# ==========================================
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("Nvidia GPU detected! Squeezing into 4-bit mode for 4GB VRAM...")

    # The ultimate 4-bit compression setup
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # Keeps the math fast
        bnb_4bit_use_double_quant=True,       # Squeezes it even smaller
        bnb_4bit_quant_type="nf4"             # Preserves AI intelligence
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config
    )
else:
    print("GPU not found. Falling back to CPU (This will be slow)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        dtype=torch.float32
    )

print(f"Model loaded successfully on {device}!\n")

# ==========================================
# 2. THE LOCAL REVIEW FUNCTION (STRICT JSON MODE)
# ==========================================
def review_code(patch_text, language="Java"):
    # We turn the AI into a strict JSON-generating machine
    system_prompt = f"""You are a strict, automated {language} code review API.
You must output your review STRICTLY in valid JSON format.
DO NOT include any conversational text, greetings, or explanations before or after the JSON.
DO NOT use markdown code blocks like ```json. Just output the raw JSON object.

Use this exact JSON schema:
{{
  "bug_found": boolean,
  "severity": "CRITICAL" | "WARNING" | "PASS",
  "issue": "One precise sentence describing the problem",
  "fixed_code": "The exact corrected Java code"
}}"""

    user_prompt = f"Code to review:\n{patch_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print(f"AI is thinking on {device}...")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,  # Keep this low so it stays robotic and predictable
        do_sample=True
    )

    input_length = inputs["input_ids"].shape[1]
    review_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return review_text.strip()

# ==========================================
# 3. THE ULTIMATE TEST
# ==========================================
sample_spring_boot_patch = """
+ public User getUser(Long id) {
+     // Fetch user from database
+     return userRepository.findById(id).get();
+ }
"""

print("--- Sending Code to AI Reviewer ---")
print(sample_spring_boot_patch)
print("--- AI Reviewer Feedback ---")
print(review_code(sample_spring_boot_patch))