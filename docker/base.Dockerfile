FROM python:3.8


ENV SHELL /bin/bash
WORKDIR /code

RUN apt-get update &&\
    apt-get install -y \
        libgomp1 \
        gcc \
        curl \
        git \
        ssh &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

COPY pyproject.toml poetry.lock /code/
RUN poetry config virtualenvs.create false &&\
    poetry install

WORKDIR /plx-context/artifacts
