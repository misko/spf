# Start with the latest Ubuntu image
FROM ubuntu:latest as base

# Prevent interactive prompts and install OS-level dependencies
# (Add or remove packages as needed for your project)
RUN apt-get update && apt-get install -y \
    python3-pip python3 python3-venv \
    git screen libiio-dev libiio-utils vim python3-dev uhubctl libusb-dev libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone your repo
RUN git clone https://github.com/misko/spf.git /spf

# Change working directory
WORKDIR /spf

RUN python3 -m venv /opt/spf-env
ENV PATH="/opt/spf-env/bin:$PATH"

COPY .env /app/.env

RUN pip install --upgrade pip
# In this stage, you can confirm everything's installed

# Install your package (which has a pyproject.toml)
# If you have a specific dev or optional dependencies, you can add them here as well
RUN pip install -e .

# Launch your tests; you can change this to 'pytest' or any other test runner
CMD ["pytest", "tests"]
