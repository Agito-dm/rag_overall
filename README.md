# rag_overall

Учебная RAG-система на Python, Gemini API и Weaviate
для ответов на вопросы по локальной базе знаний.

## Архитектура

    Documents
        -> Preprocessing
        -> Chunking
        -> Embeddings
        -> Weaviate
        -> Semantic Retrieval
        -> LLM Answer + Sources

На текущем этапе реализована подготовка документов
и их индексация в векторной базе.
Retrieval и генерация ответов находятся в разработке.

## Технологический стек

- Python 3.12
- Weaviate 1.39.17
- Docker Compose
- Gemini Embedding API
- Pydantic
- pytest
- Ruff

## Установка

    git clone https://github.com/Agito-dm/rag_overall.git
    cd rag_overall

    conda create -n rag_overall python=3.12 -y
    conda activate rag_overall

    python -m pip install -e ".[dev]"
    python -m pip check

## Конфигурация

Переменная `EMBEDDING_REQUEST_INTERVAL_SECONDS`
задаёт минимальный интервал между началами запросов embeddings.

По умолчанию используется интервал 1 секунда,
что ограничивает частоту последовательных запросов
примерно до 60 RPM в рамках одного экземпляра Embedder.

Значение `0` отключает искусственную задержку.

При необходимости интервал можно изменить в `.env`
с учётом действующих квот Gemini API.

Файл `.env` исключён из Git.

## Запуск Weaviate

Открыть Docker Desktop и убедиться,
что Docker Engine запущен.

    docker compose up -d
    docker compose ps

Проверка готовности:

    curl.exe -i http://127.0.0.1:8081/v1/.well-known/ready

HTTP API: `127.0.0.1:8081`.

gRPC: `127.0.0.1:50051`.

## Использование

Загрузка и очистка документов:

    python main.py

Демонстрация chunking:

    python main.py --chunks

Изменение параметров разбиения для экспериментов:

    python main.py --chunks --chunk-size 250 --overlap 50

Индексация базы знаний:

    python main.py --index

Повторная индексация неизменённой базы
не создаёт дубли и не требует новых embeddings.

## База знаний и индексация

Локальная база находится в `knowledge_base/`
и содержит восемь текстовых документов.

Параметры индексации:

- Chunk size: 400 символов.
- Overlap: 80 символов.
- Embedding model: gemini-embedding-001.
- Embedding dimensions: 768.
- Vector index: HNSW.
- Distance metric: cosine.
- Weaviate collection: KnowledgeChunk.

При стандартной конфигурации создаётся 25 чанков.


### Перестроение индекса

Повторный запуск индексации неизменённой базы
не создаёт дубли и не требует новых embeddings.

При изменении документов, параметров chunking,
embedding-модели или размерности векторов
необходимо перестроить коллекцию.

Для этого при запущенном Weaviate удалить
существующую коллекцию KnowledgeChunk:

    curl.exe -f -X DELETE http://127.0.0.1:8081/v1/schema/KnowledgeChunk

Внимание: команда удаляет коллекцию вместе со всеми
сохранёнными в ней объектами и векторами.

После успешного удаления выполнить:

    python main.py --index
    python -m pytest tests/integration_tests -m integration -v -s

Коллекция будет создана заново, а embeddings
будут получены для всех актуальных чанков.

При использовании нестандартных параметров chunking
интеграционный тест необходимо согласовать с ними.


## Ограничения

- Используется символьное разбиение текста.
- Индексация выполняется последовательно.
- При изменении документов, модели или параметров
  chunking необходимо перестроить коллекцию.
- Автоматическая синхронизация изменённых
  и удалённых документов не реализована.


## Тестирование

### Unit-тесты

    python -m pytest -q

По умолчанию запускаются только unit-тесты.
Для них не требуется запущенный Weaviate
или действующий Gemini API-ключ.

### Интеграционные тесты

После запуска Weaviate и индексации базы знаний:

    python -m pytest tests/integration_tests -m integration -v -s

Проверяются:

- доступность Weaviate и наличие коллекции;
- количество сохранённых объектов;
- уникальность и соответствие UUID;
- соответствие текстов и метаданных исходным чанкам;
- размерность и корректность embeddings;
- L2-норма векторов.

Интеграционные тесты не выполняют запросы к Gemini API
и не изменяют содержимое векторной базы.

### Проверка качества кода

    python -m ruff check .
    python -m ruff format --check .


## Документация

- [План проекта](docs/roadmap.md)
- [День 1 — Document Loading](docs/day01_report.md)
- [День 2 — Chunking](docs/day02_report.md)
- [День 3 — Weaviate & Embeddings](docs/day03_report.md)
