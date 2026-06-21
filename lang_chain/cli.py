"""Command-line interface for LangChain + Pinecone testing."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from lang_chain.config import load_settings
from lang_chain.services import build_index_service, build_search_service


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
