FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# install build tools and OS libs needed by OpenCV/InsightFace (libxcb, GL, etc.)
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		build-essential python3-dev g++ cmake pkg-config wget \
		libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 libxcb1 libopenblas-dev && \
	rm -rf /var/lib/apt/lists/*

# Use BuildKit cache for pip to reuse wheel/cache across builds. Requires BuildKit enabled when building.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
	pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt

COPY ./src /app/src

EXPOSE 8000

# Ensure Python can import the `api` package from the mounted source folder
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]