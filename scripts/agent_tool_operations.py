#!/usr/bin/env python3
"""Small, real blocking operations used by the agent tool-gap benchmark."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader


USER_AGENT = "AgentShift-artifact/1.0"


class TextCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_chars = 0
        self.links = 0

    def handle_data(self, data: str) -> None:
        self.text_chars += len(data.strip())

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "a":
            self.links += 1


def fetch(url: str, timeout: float, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise RuntimeError(f"response exceeds {max_bytes} bytes")
        return payload


def search(args: argparse.Namespace) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "search": args.query,
            "per-page": 5,
            "select": "id,title,publication_year",
        }
    )
    payload = json.loads(
        fetch(f"https://api.openalex.org/works?{query}", args.timeout)
    )
    results = payload.get("results", [])
    return {
        "operation": "openalex-search",
        "query": args.query,
        "results": len(results),
        "titles": [item.get("title", "")[:120] for item in results],
    }


def page(args: argparse.Namespace) -> dict[str, Any]:
    separator = "&" if "?" in args.url else "?"
    url = f"{args.url}{separator}agentshift_nonce={args.nonce}"
    payload = fetch(url, args.timeout)
    parser = TextCounter()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return {
        "operation": "page-fetch-parse",
        "url": args.url,
        "response_bytes": len(payload),
        "text_chars": parser.text_chars,
        "links": parser.links,
    }


def external_api(args: argparse.Namespace) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "UTC",
        }
    )
    payload = json.loads(
        fetch(f"https://api.open-meteo.com/v1/forecast?{query}", args.timeout)
    )
    current = payload.get("current", {})
    return {
        "operation": "open-meteo-api",
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "temperature_2m": current.get("temperature_2m"),
        "wind_speed_10m": current.get("wind_speed_10m"),
    }


def pdf(args: argparse.Namespace) -> dict[str, Any]:
    reader = PdfReader(args.path)
    pages = min(len(reader.pages), args.max_pages)
    text_chars = 0
    for index in range(pages):
        text_chars += len(reader.pages[index].extract_text() or "")
    return {
        "operation": "pdf-parse",
        "file": str(args.path),
        "file_bytes": args.path.stat().st_size,
        "document_pages": len(reader.pages),
        "parsed_pages": pages,
        "text_chars": text_chars,
    }


def python_work(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "ast":
        files = sorted(args.root.rglob("*.py"))
        nodes = 0
        source_bytes = 0
        for path in files:
            source = path.read_text(errors="replace")
            source_bytes += len(source.encode())
            nodes += sum(1 for _ in ast.walk(ast.parse(source)))
        return {
            "operation": "python-ast",
            "files": len(files),
            "source_bytes": source_bytes,
            "ast_nodes": nodes,
        }
    if args.mode == "json":
        payload = json.loads(args.json_path.read_text())
        events = payload.get("events", [])
        return {
            "operation": "python-json",
            "events": len(events),
            "context_tokens": sum(int(row.get("context_tokens", 0)) for row in events),
        }
    if args.mode == "hash":
        digest = hashlib.pbkdf2_hmac(
            "sha256", b"agentshift", b"tool-gap", args.rounds
        )
        return {
            "operation": "python-hash",
            "rounds": args.rounds,
            "digest": digest.hex(),
        }
    if args.mode == "sort":
        rng = random.Random(20260810)
        values = [rng.random() for _ in range(args.items)]
        values.sort()
        return {
            "operation": "python-sort",
            "items": args.items,
            "median": values[len(values) // 2],
        }
    raise ValueError(f"unsupported Python mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--timeout", type=float, default=15.0)
    search_parser.set_defaults(handler=search)

    page_parser = subparsers.add_parser("page")
    page_parser.add_argument("--url", required=True)
    page_parser.add_argument("--nonce", required=True)
    page_parser.add_argument("--timeout", type=float, default=15.0)
    page_parser.set_defaults(handler=page)

    api_parser = subparsers.add_parser("api")
    api_parser.add_argument("--latitude", type=float, required=True)
    api_parser.add_argument("--longitude", type=float, required=True)
    api_parser.add_argument("--timeout", type=float, default=15.0)
    api_parser.set_defaults(handler=external_api)

    pdf_parser = subparsers.add_parser("pdf")
    pdf_parser.add_argument("--path", type=Path, required=True)
    pdf_parser.add_argument("--max-pages", type=int, default=12)
    pdf_parser.set_defaults(handler=pdf)

    python_parser = subparsers.add_parser("python")
    python_parser.add_argument(
        "--mode", choices=("ast", "json", "hash", "sort"), required=True
    )
    python_parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    python_parser.add_argument(
        "--json-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/autoscaling/manifests/autoscaling-manifest-qwen8b-30m-v1.json",
    )
    python_parser.add_argument("--rounds", type=int, default=250_000)
    python_parser.add_argument("--items", type=int, default=250_000)
    python_parser.set_defaults(handler=python_work)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(args.handler(args), sort_keys=True))


if __name__ == "__main__":
    main()
