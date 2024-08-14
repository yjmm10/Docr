FROM python:3.10

LABEL name="Docr"
LABEL version="0.2.1"
LABEL description="🛠️ Docr: A modular toolkit for intelligent document analysis and processing. Easy to expand and flexible to use, just like playing with building blocks!"

WORKDIR /app

COPY . ./app

RUN poetry install

RUN pytest
# CMD ["python"]