FROM python:3.8-slim AS base

EXPOSE 8501

ENV PYTHONUNBUFFERED TRUE
RUN mkdir -p /root/.config/matplotlib &&\
  echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
WORKDIR /app
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

COPY . /app
RUN poetry config virtualenvs.create false &&\
    poetry install --without dev, test

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
