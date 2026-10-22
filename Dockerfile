FROM node:20-bookworm-slim AS client-build
WORKDIR /build/client
COPY client/package*.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM node:20-bookworm-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    DIP_URL=http://127.0.0.1:8000 \
    REVIVEAI_DISABLE_AI_SCRATCH=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY dip-service/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
ENV PATH="/opt/venv/bin:${PATH}"
COPY dip-service/ ./dip-service/
RUN mkdir -p dip-service/ai-runtime/old-photo-scratch \
    && curl -fL --retry 3 "https://huggingface.co/SkylinkLabs/restore_old_image/resolve/main/FT_Epoch_latest.pt?download=true" -o /tmp/FT_Epoch_latest.pt \
    && echo "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95  /tmp/FT_Epoch_latest.pt" | sha256sum -c - \
    && python3 dip-service/prepare_scratch_weights.py /tmp/FT_Epoch_latest.pt dip-service/ai-runtime/old-photo-scratch/scratch_model.pt \
    && rm /tmp/FT_Epoch_latest.pt
COPY server/package*.json ./server/
RUN cd server && npm ci --omit=dev
COPY server/ ./server/
COPY --from=client-build /build/client/dist ./client/dist
COPY start.mjs ./start.mjs
EXPOSE 5000
CMD ["node", "/app/start.mjs"]
