from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Iterable

from .config import EMBED_PRICE_PER_MILLION


def estimate_cost_for_files(file_paths: Iterable[str]) -> tuple[int, int, float]:
    """
    Retorna (total_chars, est_tokens, est_cost_usd).
    """
    total_chars = 0
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                total_chars += len(f.read())
        except Exception as e:
            print(f"  [warn] Não consegui ler {path} para estimativa: {e}")

    est_tokens = int(total_chars / 4.0)
    est_cost = (est_tokens / 1_000_000.0) * EMBED_PRICE_PER_MILLION
    return total_chars, est_tokens, est_cost


def show_cost_and_confirm(file_paths: list[str]) -> None:
    total_chars, est_tokens, est_cost = estimate_cost_for_files(file_paths)
    print("=== Estimativa de custo de embedding ===")
    print(f"Número total de caracteres: {total_chars:,}")
    print(f"Tokens estimados:          {est_tokens:,} (assumindo ~4 chars/token)")
    print(
        "Custo único aproximado para embedar: "
        f"US${est_cost:.4f} "
        f"(a US${EMBED_PRICE_PER_MILLION:.2f} por 1M tokens)"
    )
    print("(Estimativa grosseira; o billing real pode variar um pouco.)")
    ans = input("Prosseguir com o índice/extensão e esse custo aproximado? [y/N]: ").strip().lower()
    if ans not in ("y", "yes", "s", "sim"):
        print("Abortando operação.")
        raise SystemExit(0)


def convert_org_to_md_temp(org_path: str) -> str:
    """
    Converte um arquivo .org para .md em diretório temporário.
    Retorna o caminho do .md temporário.
    Requer 'pandoc' instalado no sistema.
    """
    if not org_path.lower().endswith(".org"):
        raise ValueError("convert_org_to_md_temp só aceita arquivos .org")

    tmp_dir = tempfile.mkdtemp(prefix="rag_org_")
    base = os.path.splitext(os.path.basename(org_path))[0]
    md_path = os.path.join(tmp_dir, base + ".md")

    try:
        subprocess.run(
            ["pandoc", org_path, "-o", md_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            "pandoc não encontrado no sistema.\n"
            "Instale o pandoc para converter arquivos .org para .md."
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Falha ao converter {org_path} para Markdown via pandoc: {e}")

    return md_path


def prepare_file_for_upload(path: str) -> tuple[str, str | None]:
    """
    Prepara um arquivo para upload.
    - Se for .org -> converte para .md temporário e retorna (upload_path, tmp_path)
    - Caso contrário -> (path, None)
    """
    if path.lower().endswith(".org"):
        tmp_md = convert_org_to_md_temp(path)
        return tmp_md, tmp_md
    return path, None
