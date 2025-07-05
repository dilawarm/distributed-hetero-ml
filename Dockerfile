FROM python:3.13-slim

WORKDIR "/distributed_hetero_ml"

ENV UV_PYTHON_PREFERENCE=managed
ENV UV_PYTHON=3.13
ENV UV_NO_PROGRESS=true
ENV PATH="/root/.local/bin:${PATH}"
ENV PATH="/root/.cargo/bin:${PATH}"

# Install system dependencies, uv, and clean up in one layer
RUN apt-get update -yq && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -q -o=Dpkg::Use-Pty=0 -y --no-install-recommends \
        pkg-config \
        libicu-dev \
        curl \
        git \
        make && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    pip install setuptools_scm && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock Makefile ./
COPY distributed_hetero_ml/ distributed_hetero_ml/

# Install Python dependencies and clean up
RUN uv run make install && \
    uv clean && \
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
