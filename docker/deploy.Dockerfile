# pull base image from dockerhub
FROM python:3.8-slim as base

# set environment variables and working direcotry for both stages
# streamlit applications cannot be run from the root directory of Linux distribution
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

# prepare the first stage for building and installing dependencies
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

RUN mkdir /app/divae_patch
COPY pyproject.toml poetry.lock ./
COPY divae_patch/divae-0.2.0-py3-none-any.whl /app/divae_patch
RUN poetry export -f requirements.txt --output requirements.txt \
    --without dev,test --with deploy
RUN /venv/bin/pip install -r requirements.txt

COPY saucie ./
RUN poetry build && /venv/bin/pip install dist/*.whl

# prepare second build stage for copying dependencies from builder
# copy the application files
# run the streamlit application
FROM base as dev

COPY --from=builder /venv /venv
COPY docker-entrypoint.sh streamlit_app.py ./
COPY saucie streamlit_elements ./
CMD ["./docker-entrypoint.sh"]
