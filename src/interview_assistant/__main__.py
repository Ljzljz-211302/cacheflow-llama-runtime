from __future__ import annotations

import argparse
from pathlib import Path

from .knowledge import KnowledgeIndex
from .llama_client import LlamaClient
from .server import create_server
from .service import InterviewService
from .store import ConversationStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CacheFlow-backed interview study assistant")
    parser.add_argument("--knowledge-root", action="append", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("runtime/interview-assistant.db"))
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-concurrent", type=int, default=8)
    return parser.parse_args()


def read_api_key(path: Path) -> str:
    keys = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not keys:
        raise ValueError("API key file must contain at least one non-empty key")
    return keys[0]


def main() -> None:
    args = parse_args()
    knowledge = KnowledgeIndex(args.knowledge_root)
    store = ConversationStore(args.db)
    model = LlamaClient(args.llama_url, read_api_key(args.api_key_file), args.model)
    server = create_server(
        args.host,
        args.port,
        InterviewService(store, knowledge, model),
        max_concurrent=args.max_concurrent,
    )
    print(
        f"Interview Assistant: http://{args.host}:{server.server_port} "
        f"({knowledge.document_count} documents, {len(knowledge.chunks)} chunks)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
