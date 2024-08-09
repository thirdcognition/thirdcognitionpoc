FROM python:3.12.5-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY poctimeline poctimeline

RUN pip install -r requirements.txt
RUN mkdir -p db

EXPOSE 3500
EXPOSE 4000
HEALTHCHECK CMD curl --fail http://localhost:3500/_stcore/health
ENTRYPOINT ["start.sh"]