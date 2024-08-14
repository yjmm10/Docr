FROM python:3.10

LABEL name="Docly"
LABEL version="0.0.3"
LABEL description="🛠️ Docly: A modular toolkit for intelligent document analysis and processing. Easy to expand and flexible to use, just like playing with building blocks!"

WORKDIR /app

COPY . ./app

RUN poetry install

RUN pytest
# CMD ["python"]