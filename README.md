# Pinecone Example CLI

Простой CLI-сервис на Python для тестирования работы с векторной БД
[Pinecone](https://www.pinecone.io/). Эмбеддинги создаются через
[ProxyAPI](https://proxyapi.ru/) с моделью OpenAI `text-embedding-3-small`.

## Возможности

- **index** — создание serverless-индекса (если ещё нет) и загрузка
  документов с эмбеддингами
- **search** — семантический поиск по индексу

Источник данных: одна строка (`--text`) или файл `.txt` / `.json`
(`--file`).

## Требования

- Python 3.11+
- API-ключ [Pinecone](https://app.pinecone.io/)
- API-ключ [ProxyAPI](https://proxyapi.ru/)

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Или через Makefile (если настроен `pipi`):

```bash
make venv
make install
```

## Настройка

Скопируйте шаблон и заполните ключи:

```bash
cp .env.example .env
```

| Переменная           | Описание                          |
| -------------------- | --------------------------------- |
| `PINECONE_API_KEY`   | Ключ Pinecone                     |
| `PROXYAPI_API_KEY`   | Ключ ProxyAPI                     |
| `PINECONE_CLOUD`     | Облако (по умолчанию `aws`)       |
| `PINECONE_REGION`    | Регион (по умолчанию `us-east-1`) |

## Использование

### Индексация

Одна строка:

```bash
python -m pinecone_cli index \
  --name demo-index \
  --text "Canon выпускает зеркальные и беззеркальные камеры"
```

Из текстового файла (одна строка — один документ, пустые пропускаются):

```bash
python -m pinecone_cli index \
  --name cameras \
  --file data/canon-nikon-phrases.txt
```

Из JSON (массив строк или объектов с полем `text` / `content`):

```bash
python -m pinecone_cli index \
  --name demo-index \
  --file etc/sample.json
```

Опционально — namespace:

```bash
python -m pinecone_cli index \
  --name demo-index \
  --file etc/sample.txt \
  --namespace my-ns
```

### Поиск

```bash
python -m pinecone_cli search \
  --name cameras \
  --query "когда основана компания Canon" \
  --top-k 5
```

### Makefile

```bash
make cli-index                          # etc/sample.txt → demo-index
make cli-search                         # поиск в demo-index

make cli-index INDEX=cameras            # свой индекс
make cli-search INDEX=cameras           # поиск в cameras
```

Для индексации `data/canon-nikon-phrases.txt`:

```bash
python -m pinecone_cli index \
  --name cameras \
  --file data/canon-nikon-phrases.txt

python -m pinecone_cli search \
  --name cameras \
  --query "история Nikon" \
  --top-k 3
```

## Структура проекта

```
pinecone_cli/
  config.py          # настройки из .env
  loaders.py         # загрузка текста и файлов
  embedding.py       # эмбеддинги через ProxyAPI
  pinecone_store.py  # операции с Pinecone
  services.py        # оркестрация index / search
  cli.py             # команды CLI
data/                # примеры данных
etc/                 # sample-файлы и заметки
```

## Технические детали

- Модель эмбеддингов: `text-embedding-3-small` (1536 измерений)
- ProxyAPI endpoint: `https://api.proxyapi.ru/openai/v1`
- Метрика индекса: cosine similarity
- Тип индекса: Pinecone serverless
