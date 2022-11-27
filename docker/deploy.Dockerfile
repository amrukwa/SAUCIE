FROM python:3.8-slim as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.2.0

RUN apt-get update &&\
    apt-get install -y \
        gcc &&\
    rm -rf /var/lib/apt/lists/*
RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt \
    --without dev,test --with deploy
RUN /venv/bin/pip install -r requirements.txt

COPY saucie ./
RUN poetry build && /venv/bin/pip install dist/*.whl

FROM base as dev

COPY --from=builder /venv /venv
COPY docker-entrypoint.sh streamlit_app.py ./
COPY saucie streamlit_elements ./
CMD ["./docker-entrypoint.sh"]
