FROM python@sha256:09a17cad02668ae61ece136b4fd36ac3849e14f670ebedb637ea3e15580525bc AS deps
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HOME=/tmp
COPY requirements-onnx.txt /tmp/requirements-onnx.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-onnx.txt

FROM deps AS runtime
WORKDIR /app
COPY bundle/ /app/
USER 65532:65532
ENTRYPOINT ["python", "/app/runner.py", "--bundle", "/app", "--runtime", "onnx"]
