from __future__ import annotations

import json
import os
from typing import Any

from .config import VECTOR_STORE_CACHE_PATH


def _empty_cache() -> dict[str, Any]:
    return {
        "sessions": {},       # key -> vs_id
        "files_per_vs": {},   # vs_id -> [abs_path, ...]
        "_last": None,        # último vs_id usado
    }


def load_cache() -> dict[str, Any]:
    path = VECTOR_STORE_CACHE_PATH
    if not path.exists():
        return _empty_cache()

    try:
        raw = json.loads(path.read_text())
    except Exception:
        return _empty_cache()

    # Formato novo
    if isinstance(raw, dict) and ("sessions" in raw or "files_per_vs" in raw):
        cache = _empty_cache()
        cache["sessions"] = raw.get("sessions", {})
        cache["files_per_vs"] = raw.get("files_per_vs", {})
        cache["_last"] = raw.get("_last")
        return cache

    # Formato antigo: { "/path": "vs_...", "_last": "vs_..." }
    cache = _empty_cache()
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        cache["sessions"][k] = v
    cache["_last"] = raw.get("_last")
    return cache


def save_cache(cache: dict[str, Any]) -> None:
    base = _empty_cache()
    base.update(cache or {})
    VECTOR_STORE_CACHE_PATH.write_text(json.dumps(base, indent=2))


def canonical_key(path_or_glob: str) -> str:
    """
    Normaliza um diretório ou glob para ser usado como chave de sessão.
    Diretórios -> caminho absoluto; globs -> string expandida.
    """
    expanded = os.path.expanduser(os.path.expandvars(path_or_glob))
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)
    return expanded


def save_vector_store_id_for_key(key: str, vs_id: str) -> None:
    cache = load_cache()
    cache["sessions"][key] = vs_id
    cache["_last"] = vs_id
    save_cache(cache)


def load_vector_store_id_for_key(key: str) -> str | None:
    cache = load_cache()
    return cache["sessions"].get(key)


def load_last_vector_store_id() -> str | None:
    cache = load_cache()
    return cache.get("_last")


def add_indexed_files(vs_id: str, files: list[str]) -> None:
    """
    Adiciona arquivos (paths absolutos) como já indexados para um vs_id.
    """
    cache = load_cache()
    fps = cache.setdefault("files_per_vs", {})
    existing = set(fps.get(vs_id, []))
    new_abs = {os.path.abspath(p) for p in files}
    combined = sorted(existing.union(new_abs))
    fps[vs_id] = combined
    save_cache(cache)


def get_indexed_files(vs_id: str) -> set[str]:
    cache = load_cache()
    fps = cache.get("files_per_vs", {})
    return set(fps.get(vs_id, []))


def list_vector_stores() -> None:
    cache = load_cache()
    sessions: dict[str, str] = cache.get("sessions", {})
    if not sessions:
        print("No cached vector stores.")
        return

    last_vs = cache.get("_last")
    print(f"Cache file: {VECTOR_STORE_CACHE_PATH}\n")
    print("Cached vector stores (per directory/glob key):")
    for key, vs_id in sessions.items():
        marker = "  (last used)" if vs_id == last_vs else ""
        print(f"- {key} -> {vs_id}{marker}")


def resolve_vector_store_id(arg: str) -> str:
    """
    Resolve um argumento (auto, vs_..., path/glob) para um vector_store_id real.
    """
    # auto -> último
    if arg == "auto":
        vs = load_last_vector_store_id()
        if not vs:
            raise RuntimeError(
                "Nenhum vector store em cache.\n"
                "Rode antes:\n"
                "  rag-cli index PATH_OR_GLOB"
            )
        return vs

    # id explícito
    if arg.startswith("vs_"):
        return arg

    # path/glob
    key = canonical_key(arg)
    vs = load_vector_store_id_for_key(key)
    if not vs:
        raise RuntimeError(
            "Nenhum vector store em cache para essa chave:\n"
            f"  {key}\n"
            "Faça o índice antes com:\n"
            f"  rag-cli index \"{arg}\""
        )
    return vs
