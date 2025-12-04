from __future__ import annotations

import sys

from .cache import (
    canonical_key,
    save_vector_store_id_for_key,
    resolve_vector_store_id,
    list_vector_stores,
)
from .rag import index_path_or_glob, extend_path_or_glob, ask, chat
from .config import VECTOR_STORE_CACHE_PATH, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT


def print_usage_and_exit() -> None:
    print("Uso:")
    print("  rag-cli index PATH_OR_GLOB")
    print("  rag-cli extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB")
    print("  rag-cli ask [--model MODEL] [--system PROMPT] VECTOR_STORE_ID \"sua pergunta\"")
    print("  rag-cli ask [--model MODEL] [--system PROMPT] auto \"sua pergunta\"")
    print("  rag-cli ask [--model MODEL] [--system PROMPT] PATH_OR_GLOB \"sua pergunta\"")
    print("  rag-cli chat [--model MODEL] [--system PROMPT] auto")
    print("  rag-cli chat [--model MODEL] [--system PROMPT] PATH_OR_GLOB")
    print("  rag-cli chat [--model MODEL] [--system PROMPT] VECTOR_STORE_ID")
    print("  rag-cli list                             # lista índices cacheados")
    print()
    print(f"  Modelo padrão: {DEFAULT_MODEL}")
    if DEFAULT_SYSTEM_PROMPT:
        print(f"  System prompt padrão (RAG_CLI_SYSTEM): {DEFAULT_SYSTEM_PROMPT!r}")
    sys.exit(1)


def _parse_model_and_system(args: list[str]) -> tuple[list[str], str | None, str | None]:
    """
    Faz um parsing bem simples de flags:
      [--model MODEL] [--system PROMPT] RESTO...

    Retorna (resto_args, model, system_prompt).
    Não faz nada especial com outras flags (que não existem hoje).
    """
    model: str | None = None
    system: str | None = None
    i = 0
    out: list[str] = []

    while i < len(args):
        a = args[i]
        if a in ("-m", "--model"):
            if i + 1 >= len(args):
                raise SystemExit("Erro: --model requer um argumento.")
            model = args[i + 1]
            i += 2
        elif a in ("-s", "--system"):
            if i + 1 >= len(args):
                raise SystemExit("Erro: --system requer um argumento.")
            system = args[i + 1]
            i += 2
        else:
            out.append(a)
            i += 1

    return out, model, system


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
        # Sintaxe suportada:
        #   rag-cli ask [--model M] [--system S] TARGET palavras da pergunta...
        # onde TARGET = auto | vs_... | PATH_OR_GLOB (inclusive qualquer coisa que seu resolve_vector_store_id trate, ex: fzf)
        if len(argv) < 4:
            print("Uso: rag-cli ask [--model MODEL] [--system PROMPT] TARGET \"sua pergunta\"")
            sys.exit(1)

        # fazemos parsing das flags a partir de argv[2:]
        tail, model_flag, system_flag = _parse_model_and_system(argv[2:])
        if len(tail) < 2:
            print("Uso: rag-cli ask [--model MODEL] [--system PROMPT] TARGET \"sua pergunta\"")
            sys.exit(1)

        arg2 = tail[0]        # TARGET (auto, vs_..., PATH_OR_GLOB, ou o que seu resolve_... aceitar, inclusive fzf)
        question_words = tail[1:]
        vector_store_id = resolve_vector_store_id(arg2)
        question = " ".join(question_words)

        # se system_flag for None, ask() vai usar DEFAULT_SYSTEM_PROMPT internamente
        answer = ask(vector_store_id, question, model=model_flag, system_prompt=system_flag)
        print(answer)

    elif cmd == "chat":
        # Sintaxe suportada:
        #   rag-cli chat [--model M] [--system S] TARGET
        if len(argv) < 3:
            print("Uso: rag-cli chat [--model MODEL] [--system PROMPT] TARGET")
            sys.exit(1)

        tail, model_flag, system_flag = _parse_model_and_system(argv[2:])
        if len(tail) != 1:
            print("Uso: rag-cli chat [--model MODEL] [--system PROMPT] TARGET")
            sys.exit(1)

        arg2 = tail[0]   # TARGET
        vector_store_id = resolve_vector_store_id(arg2)
        chat(vector_store_id, model=model_flag, system_prompt=system_flag)

    elif cmd == "list":
        list_vector_stores()

    else:
        print(f"Comando desconhecido: {cmd}")
        print("Use 'index', 'extend', 'ask', 'chat' ou 'list'.")
        sys.exit(1)
