import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. DOWNLOAD & LOAD THE QWEN BRAIN
# ==========================================
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

print(f"Loading {model_name}... (If this is your first time, it will download ~6.5GB)")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if we can use a local Nvidia GPU, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# We load it using standard precision for CPU compatibility
# CHANGED: 'torch_dtype' is now 'dtype'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    dtype=torch.float32 if device == "cpu" else torch.float16
)
print(f"Model loaded successfully on {device}!\n")


# ==========================================
# 2. THE LOCAL REVIEW FUNCTION
# ==========================================
def review_code(patch_text, language="Java"):
    system_prompt = f"""You are a ruthless, senior {language} backend engineer.
Your job is to review the following code patch.
DO NOT say hello. DO NOT write an essay.
If there is a bug, point it out immediately and provide the exact code fix.
Pay special attention to NullPointerExceptions and database inefficiencies."""

    user_prompt = f"Code to review:\n{patch_text}"

    # Qwen natively understands this ChatML format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Format the prompt and send it to the CPU/GPU
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print("AI is thinking... (This may take 15-30 seconds on a CPU)")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True
    )

    # Slice off the prompt to just get the review
    input_length = inputs["input_ids"].shape[1]
    review_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return review_text.strip()


# ==========================================
# 3. THE ULTIMATE TEST
# ==========================================
# Our dangerous Spring Boot code
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