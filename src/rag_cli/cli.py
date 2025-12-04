from __future__ import annotations

import sys

from .cache import (
    canonical_key,
    save_vector_store_id_for_key,
    resolve_vector_store_id,
    list_vector_stores,
)
from .rag import index_path_or_glob, extend_path_or_glob, ask, chat
from .config import VECTOR_STORE_CACHE_PATH


def print_usage_and_exit() -> None:
    print("Uso:")
    print("  rag-cli index PATH_OR_GLOB")
    print("  rag-cli extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB")
    print("  rag-cli ask VECTOR_STORE_ID \"sua pergunta\"")
    print("  rag-cli ask auto \"sua pergunta\"          # último índice")
    print("  rag-cli ask PATH_OR_GLOB \"sua pergunta\"  # índice por diretório/glob")
    print("  rag-cli chat auto                        # chat interativo com último índice")
    print("  rag-cli chat PATH_OR_GLOB                # chat interativo com índice daquele path")
    print("  rag-cli list                             # lista índices cacheados")
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print_usage_and_exit()

    cmd = argv[1]

    if cmd == "index":
        if len(argv) != 3:
            print("Uso: rag-cli index PATH_OR_GLOB")
            sys.exit(1)
        pattern = argv[2]
        vs_id = index_path_or_glob(pattern)
        key = canonical_key(pattern)
        # index_path_or_glob já salva, mas garantimos:
        save_vector_store_id_for_key(key, vs_id)
        print("\n=== VECTOR_STORE_ID (cacheado) ===")
        print(f"Key: {key}")
        print(f"ID : {vs_id}")
        print(f"Salvo em: {VECTOR_STORE_CACHE_PATH}")

    elif cmd == "extend":
        if len(argv) != 4:
            print("Uso:")
            print("  rag-cli extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB")
            print("Exemplo:")
            print("  rag-cli extend \"$HOME/mdroam\" \"$HOME/mdroam/novas/**/*.org\"")
            sys.exit(1)

        session_key = argv[2]
        new_pattern = argv[3]

        vector_store_id = resolve_vector_store_id(session_key)
        extend_path_or_glob(vector_store_id, new_pattern)

        key = canonical_key(session_key)
        save_vector_store_id_for_key(key, vector_store_id)

    elif cmd == "ask":
        if len(argv) < 4:
            print("Uso: rag-cli ask VECTOR_STORE_ID \"sua pergunta\"")
            print("     rag-cli ask auto \"sua pergunta\"")
            print("     rag-cli ask PATH_OR_GLOB \"sua pergunta\"")
            sys.exit(1)

        arg2 = argv[2]
        vector_store_id = resolve_vector_store_id(arg2)
        question = " ".join(argv[3:])
        answer = ask(vector_store_id, question)
        print(answer)

    elif cmd == "chat":
        if len(argv) < 3:
            print("Uso: rag-cli chat auto")
            print("     rag-cli chat PATH_OR_GLOB")
            print("     rag-cli chat VECTOR_STORE_ID")
            sys.exit(1)

        arg2 = argv[2]
        vector_store_id = resolve_vector_store_id(arg2)
        chat(vector_store_id)

    elif cmd == "list":
        list_vector_stores()

    else:
        print(f"Comando desconhecido: {cmd}")
        print("Use 'index', 'extend', 'ask', 'chat' ou 'list'.")
        sys.exit(1)
