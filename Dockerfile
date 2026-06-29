FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

CMD ["python", "main.py"]
