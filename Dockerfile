FROM python:3.12-slim-bookworm
RUN pip install uv

WORKDIR /app

ADD uv.lock uv.lock
ADD pyproject.toml pyproject.toml

RUN uv sync --group web --frozen

COPY routers ./routers
COPY EmbeddingsLib.py kge_model_loader.py main.py pydantic_models.py ./

EXPOSE 8000

CMD ["uv","run","uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
