FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

COPY requirements.txt /code/requirements.txt
COPY requirements-full.txt /code/requirements-full.txt
COPY requirements-embeddings.txt /code/requirements-embeddings.txt
COPY requirements-parsing.txt /code/requirements-parsing.txt
COPY requirements-ocr.txt /code/requirements-ocr.txt
COPY requirements-research-heavy.txt /code/requirements-research-heavy.txt

ARG INSTALL_FULL=false
ARG EXTRA_REQUIREMENTS=requirements.txt
RUN pip install --upgrade pip \
    && if [ "$INSTALL_FULL" = "true" ]; then \
        pip install -r /code/requirements-full.txt; \
    else \
        pip install -r /code/$EXTRA_REQUIREMENTS; \
    fi

COPY . /code/hackathon_project
WORKDIR /code

EXPOSE 8000
CMD ["uvicorn", "hackathon_project.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
