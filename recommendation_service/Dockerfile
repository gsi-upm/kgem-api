FROM python:3.12-slim-bookworm
RUN pip install uv

WORKDIR /app

ADD uv.lock uv.lock
ADD pyproject.toml pyproject.toml

RUN uv sync

COPY api ./api
COPY core ./core
COPY model_loader.py main.py pydantic_models.py ./

EXPOSE 8002

CMD ["uv","run","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
