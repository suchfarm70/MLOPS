# 1. Use NVIDIA CUDA image (Development version often needed for compiling bitsandbytes)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2. Set environment variables to keep Python quiet and clean
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Install Python, pip, and Git
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Set up the working directory
WORKDIR /app

# 5. Install dependencies first (for faster re-builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your code and your CUSTOM ADAPTER folder
# This will include: api.py, local_qwen_reviewer.py, and my-custom-qwen-java-reviewer/
COPY . .

# 7. Expose the FastAPI port
EXPOSE 8000

# 8. Run the Uvicorn server
# We use 'python3 -m' to ensure we use the correct environment
CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]