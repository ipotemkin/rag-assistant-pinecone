"""Command-line interface for LangChain + Pinecone testing."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from lang_chain.chat import ChatService, build_chat_service, is_exit_command
from lang_chain.config import load_settings
from lang_chain.console import ConsoleReader
from lang_chain.services import build_index_service, build_search_service
from lang_chain.url_loader import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    UrlDocumentLoader,
)


@click.group()
def cli() -> None:
    """CLI for indexing and searching Pinecone via LangChain."""


@cli.command("index")
@click.option(
    "--name",
    "index_name",
    required=True,
    help="Pinecone index name.",
)
@click.option(
    "--text",
    "text",
    default=None,
    help="Single text document to index.",
)
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to .txt or .json file with documents.",
)
@click.option(
    "--namespace",
    default="",
    show_default=True,
    help="Optional Pinecone namespace.",
)
def index_command(
    index_name: str,
    text: str | None,
    file_path: Path | None,
    namespace: str,
) -> None:
    """Create index (if needed) and upsert embedded documents."""
    _validate_index_input(text, file_path)
    try:
        settings = load_settings()
        service = build_index_service(settings)
        if text is not None:
            count = service.index_from_text(index_name, text, namespace)
        else:
            count = service.index_from_file(
                index_name,
                file_path,  # type: ignore[arg-type]
                namespace,
            )
    except (ValueError, FileNotFoundError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        f"Indexed {count} document(s) into '{index_name}' "
        f"(namespace='{namespace or 'default'}')."
    )


@cli.command("index-url")
@click.option(
    "--name",
    "index_name",
    required=True,
    help="Pinecone index name.",
)
@click.option(
    "--url",
    required=True,
    help="Web page URL to fetch, chunk, and index.",
)
@click.option(
    "--namespace",
    default="",
    show_default=True,
    help="Optional Pinecone namespace.",
)
@click.option(
    "--chunk-size",
    default=DEFAULT_CHUNK_SIZE,
    show_default=True,
    help="Maximum chunk size in characters.",
)
@click.option(
    "--chunk-overlap",
    default=DEFAULT_CHUNK_OVERLAP,
    show_default=True,
    help="Overlap between consecutive chunks.",
)
def index_url_command(
    index_name: str,
    url: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Fetch a web page, split it into chunks, and upsert embeddings."""
    try:
        settings = load_settings()
        service = build_index_service(
            settings,
            url_loader=UrlDocumentLoader(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
        )
        count = service.index_from_url(index_name, url, namespace)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        f"Indexed {count} chunk(s) from '{url}' into '{index_name}' "
        f"(namespace='{namespace or 'default'}')."
    )


@cli.command("search")
@click.option(
    "--name",
    "index_name",
    required=True,
    help="Pinecone index name.",
)
@click.option(
    "--query",
    required=True,
    help="Search query text.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of results to return.",
)
@click.option(
    "--namespace",
    default="",
    show_default=True,
    help="Optional Pinecone namespace.",
)
def search_command(
    index_name: str,
    query: str,
    top_k: int,
    namespace: str,
) -> None:
    """Search for relevant documents in the index."""
    try:
        settings = load_settings()
        service = build_search_service(settings)
        results = service.search(
            index_name=index_name,
            query=query,
            top_k=top_k,
            namespace=namespace,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    if not results:
        click.echo("No matches found.")
        return

    for rank, result in enumerate(results, start=1):
        click.echo(f"{rank}. score={result.score:.4f}")
        click.echo(f"   id={result.vector_id}")
        click.echo(f"   text={result.text}")


@cli.command("chat")
@click.option(
    "--name",
    "index_name",
    required=True,
    help="Pinecone index name.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of context documents to retrieve.",
)
@click.option(
    "--namespace",
    default="",
    show_default=True,
    help="Optional Pinecone namespace.",
)
def chat_command(
    index_name: str,
    top_k: int,
    namespace: str,
) -> None:
    """Ask questions in an interactive RAG dialog."""
    try:
        settings = load_settings()
        service = build_chat_service(
            settings,
            index_name=index_name,
            top_k=top_k,
            namespace=namespace,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        f"Chat with index '{index_name}'. "
        "Type 'exit' or 'quit' to leave."
    )
    _run_chat_loop(service)


def _run_chat_loop(service: ChatService) -> None:
    reader = ConsoleReader()
    while True:
        try:
            question = reader.read_line("You> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye.")
            reader.save_history()
            break

        if not question.strip():
            reader.drop_last_history_entry()
            continue

        if is_exit_command(question):
            click.echo("Bye.")
            reader.save_history()
            break

        try:
            answer = service.ask(question)
        except ValueError as error:
            raise click.ClickException(str(error)) from error

        click.echo(f"Assistant: {answer}\n")


def _validate_index_input(
    text: str | None,
    file_path: Path | None,
) -> None:
    if bool(text) == bool(file_path):
        raise click.ClickException(
            "Provide exactly one source: --text or --file."
        )


def main() -> None:
    """CLI entry point."""
    try:
        cli(prog_name="lang-chain")
    except click.ClickException as error:
        click.echo(f"Error: {error.message}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
