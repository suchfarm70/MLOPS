
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel  # <--- NEW MLOPS IMPORT!

# ==========================================
# 1. DOWNLOAD & LOAD THE QWEN BRAIN (GPU OPTIMIZED)
# ==========================================
base_model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
adapter_path = "./my-custom-qwen-java-reviewer/my-custom-qwen-java-reviewer"

print(f"Loading Tokenizer from {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("Nvidia GPU detected! Squeezing into 4-bit mode for 4GB VRAM...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Step A: Load the heavy Base Model
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=quantization_config
    )

    # Step B: Snap your custom trained Adapter on top!
    print("Attaching Custom LoRA Brain...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        autocast_adapter_dtype=False  # <--- THE MAGIC BYPASS
    )
else:
    print("GPU not found. Falling back to CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",
        dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

print(f"Custom Model successfully assembled and loaded on {device}!\n")
# ==========================================
# 2. THE LOCAL REVIEW FUNCTION (STRICT JSON MODE)
# ==========================================
def review_code(patch_text, language="Java"):
    # We switch to XML tags to completely bypass JSON escaping crashes
    system_prompt = f"""You are a strict, automated {language} code review API.
Do NOT output JSON. You must output your review using these EXACT tags:

<bug_found>true or false</bug_found>
<severity>CRITICAL or WARNING or PASS</severity>
<issue>One precise sentence describing the problem</issue>
<fixed_code>
The exact corrected Java code goes here. Do not escape quotes.
</fixed_code>"""

    user_prompt = f"Code to review:\n{patch_text}"

    # ... (Keep the rest of the function exactly the same)

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

# Startup complete. Waiting for FastAPI to take over.