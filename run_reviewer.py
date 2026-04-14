import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ==========================================
# 1. LOAD THE LOCAL BRAIN
# ==========================================
model_path = "./checkpoint-11712"

print(f"Loading the 250M-parameter brain from {model_path}...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# If your laptop has an Nvidia GPU, it uses it. Otherwise, it safely falls back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded successfully on {device}!\n")


# ==========================================
# 2. THE REVIEW FUNCTION (TUNED FOR LOGIC)
# ==========================================
def review_code(patch_text, language="Java"):
    # We explicitly tell the model what language it is looking at
    prompt = f"Review this {language} patch and provide feedback:\n{patch_text}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Tuned for strict, logical code review
    outputs = model.generate(
        **inputs,
        max_length=256,
        do_sample=True,
        temperature=0.2,         # Cold, logical, precise
        top_p=0.9,
        repetition_penalty=1.1,
        early_stopping=True
    )

    review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return review


# ==========================================
# 3. THE REAL TEST: FIND THE BUG (WITH DIFF MARKERS)
# ==========================================
# We are simulating a Pull Request where a developer is adding a dangerous method.
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