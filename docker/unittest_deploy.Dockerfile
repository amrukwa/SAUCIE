FROM python:3.8-slim as base

ENV SHELL /bin/bash
WORKDIR /code
RUN apt-get update &&\
    apt-get install -y \
        curl \
        git \
        ssh &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

COPY pyproject.toml poetry.lock /code/
RUN mkdir /code/divae_patch
COPY divae_patch/divae-0.2.0-py3-none-any.whl /code/divae_patch
RUN poetry config virtualenvs.create false &&\
    poetry install --without deploy
COPY saucie/ ./saucie/
COPY streamlit_elements/ ./streamlit_elements/
COPY test/ ./test/
RUN poetry run pytest
