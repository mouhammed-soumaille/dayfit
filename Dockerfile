# DAYFIT: one image for both the API and the Streamlit page (see docker-compose.yml).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache
WORKDIR /app

# CPU-only PyTorch (much smaller than the GPU build; deployment latency is measured on CPU)
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what the app needs (.dockerignore is an allow-list: no results, report, LookBench or photos)
COPY common/ common/
COPY evaluation/ evaluation/
COPY retrieval/ retrieval/
COPY serving/ serving/
COPY data/catalog.csv data/catalog.csv
COPY data/images_hires/ data/images_hires/
COPY models/ models/
# the index cache checks the checkpoint's modification time, which copying into the image changes;
# restore the time recorded in the index file so the index built on the Mac is reused (not rebuilt)
RUN python -c "import os; t = float(open('models/index/finetuned_random_paths.txt').readline().split()[-1]); os.utime('models/finetuned_random.pt', (t, t))"

# warm-up at BUILD time: downloads the pretrained weights into the image and loads the saved
# catalogue index (if it could not be reused, it is rebuilt here once, not at every start)
RUN python -c "import serving.api"

# run as a normal user, not root
RUN useradd --create-home dayfit && chown -R dayfit /app
USER dayfit

EXPOSE 8000 8501
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
