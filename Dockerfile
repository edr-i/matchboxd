FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py letterboxd_recommender.py ./
COPY data/ ./data/

ENV RATINGS_CSV=""
ENV MOVIE_DATA_CSV="/app/data/movie_data.csv"
ENV COMMUNITY_CACHE="/app/data/cache_community.pkl"
ENV SAMPLE_SIZE="0"
ENV MIN_RATINGS="200"
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]