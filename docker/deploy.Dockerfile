# pull base image from dockerhub
FROM python:3.8-slim as base

# set common env variables and working directory
ENV PYTHONUNBUFFERED=1

WORKDIR /app

FROM base AS builder
RUN apt-get update &&\
    apt-get install -y \
        gcc &&\
    rm -rf /var/lib/apt/lists/*
ENV POETRY_HOME="/opt/poetry"
RUN pip install "poetry==1.2.0"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

FROM builder AS deps_builder
COPY . /app
RUN poetry install &&\
    poetry build
RUN poetry export -f requirements.txt --output requirements.txt \
    --without dev,test --with deploy

FROM builder AS deps_install
COPY --from=deps_builder /app/dist /app/dist
COPY --from=deps_builder /app/requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    --prefix=/install \
    --no-cache-dir \
    --no-warn-script-location \
    --ignore-installed &&\
    pip install /app/dist/saucie*.whl \
    --prefix=/install \
    --no-cache-dir \
    --no-warn-script-location

FROM base AS deploy
EXPOSE 8501
COPY --from=deps_install /install /usr/local
ENV PYTHONPATH="/app"
WORKDIR /app
COPY streamlit_elements/ ./streamlit_elements/
COPY ./streamlit_app.py .
CMD ["streamlit", "run", "streamlit_app.py", " --server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.headless=true"]
