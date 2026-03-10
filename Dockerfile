# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
# SPDX-License-Identifier: MIT
# UMST Prototype 2a — Reproducibility Container
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cd prototype/src/rust && cargo build --release

CMD ["bash", "-c", "python --version && cargo --version && echo 'UMST environment ready'"]
