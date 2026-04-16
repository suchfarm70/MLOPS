# ==========================================
# STAGE 1: The Builder (The Heavy Lifting)
# ==========================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Prevent stuck prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies to a specific folder we can move later
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==========================================
# STAGE 2: The Runtime (The Slim Result)
# ==========================================
# We switch from 'devel' (7GB+) to 'runtime' (much smaller)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.10/site-packages:/install/local/lib/python3.10/dist-packages:/install/lib/python3.10/dist-packages:/install/lib/python3/dist-packages:$PYTHONPATH"

# Install ONLY the bare minimum Python runner
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY the installed libraries from the builder stage
COPY --from=builder /install /install

# Copy your application code and your LoRA adapter
COPY . .

# Expose and Run
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]