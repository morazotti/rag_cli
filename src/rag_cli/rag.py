from __future__ import annotations

import os
from glob import glob
from typing import List

from .client import client, BadRequestError
from .config import SUPPORTED_EXTENSIONS, DEFAULT_MODEL
from .cache import (
    canonical_key,
    save_vector_store_id_for_key,
    add_indexed_files,
    get_indexed_files,
)
from .files import show_cost_and_confirm, prepare_file_for_upload


def _collect_supported_files(path_or_glob: str) -> tuple[list[str], list[str]]:
    """
    Retorna (file_paths_suportados_absolutos, ignorados).
    """
    expanded = os.path.expanduser(os.path.expandvars(path_or_glob))

    if os.path.isdir(expanded):
        search_pattern = os.path.join(os.path.abspath(expanded), "**", "*")
        paths = glob(search_pattern, recursive=True)
    else:
        paths = glob(expanded, recursive=True)

    file_paths: List[str] = []
    skipped: List[str] = []

    for p in paths:
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            file_paths.append(os.path.abspath(p))
        else:
            skipped.append(p)

    # remove duplicados
    file_paths = sorted(set(file_paths))
    return file_paths, skipped


def index_path_or_glob(path_or_glob: str) -> str:
    """
    Cria um vector store novo para PATH_OR_GLOB e indexa
    todos os arquivos suportados.
    Retorna o vector_store_id.
    """
    file_paths, skipped = _collect_supported_files(path_or_glob)

    if not file_paths:
        raise RuntimeError(
            f"Nenhum arquivo suportado encontrado para: {path_or_glob}\n"
            f"Extensões suportadas: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if skipped:
        print("Ignorando arquivos com extensão não suportada para retrieval:")
        for p in skipped:
            print("  ", p)
        print()

    print(f"Encontrados {len(file_paths)} arquivos suportados.")
    show_cost_and_confirm(file_paths)

    vs = client.vector_stores.create(name="rag-store-from-path")
    vector_store_id = vs.id

    print(f"\nVector store criado: {vector_store_id}")
    print(f"Enviando {len(file_paths)} arquivos...")

    indexed_success: list[str] = []

    for path in file_paths:
        tmp_path = None
        try:
            upload_path, tmp_path = prepare_file_for_upload(path)

            with open(upload_path, "rb") as fp:
                file_obj = client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id,
                    file=fp,
                )
            print(f"  OK: {path} -> file_id={file_obj.id}")
            indexed_success.append(path)
        except BadRequestError as e:
            print(f"  ERRO (400) para {path}: {e}")
        except Exception as e:
            print(f"  ERRO para {path}: {e}")
        finally:
            if tmp_path:
                import shutil

                try:
                    shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)
                except Exception:
                    pass

    if indexed_success:
        add_indexed_files(vector_store_id, indexed_success)

    print("Indexação concluída.")
    # salva sessão no cache
    key = canonical_key(path_or_glob)
    save_vector_store_id_for_key(key, vector_store_id)
    return vector_store_id


def extend_path_or_glob(vector_store_id: str, path_or_glob: str) -> None:
    """
    Anexa arquivos novos a um vector_store existente.
    NÃO reindexa arquivos já enviados para esse vector_store_id.
    """
    file_paths, skipped = _collect_supported_files(path_or_glob)

    if not file_paths:
        raise RuntimeError(
            f"Nenhum arquivo suportado encontrado para: {path_or_glob}\n"
            f"Extensões suportadas: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if skipped:
        print("Ignorando arquivos com extensão não suportada para retrieval:")
        for p in skipped:
            print("  ", p)
        print()

    already = get_indexed_files(vector_store_id)
    new_files = [p for p in file_paths if p not in already]

    if not new_files:
        print("Nenhum arquivo novo para indexar; todos já estão neste vector store.")
        return

    print(f"Encontrados {len(new_files)} arquivos NOVOS para anexar.")
    show_cost_and_confirm(new_files)

    print(f"\nAnexando arquivos ao vector store existente: {vector_store_id}")
    indexed_success: list[str] = []

    for path in new_files:
        tmp_path = None
        try:
            upload_path, tmp_path = prepare_file_for_upload(path)

            with open(upload_path, "rb") as fp:
                file_obj = client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id,
                    file=fp,
                )
            print(f"  OK: {path} -> file_id={file_obj.id}")
            indexed_success.append(path)
        except BadRequestError as e:
            print(f"  ERRO (400) para {path}: {e}")
        except Exception as e:
            print(f"  ERRO para {path}: {e}")
        finally:
            if tmp_path:
                import shutil

                try:
                    shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)
                except Exception:
                    pass

    if indexed_success:
        add_indexed_files(vector_store_id, indexed_success)

    print("Extensão concluída.")


def ask(vector_store_id: str, question: str) -> str:
    """
    Pergunta única usando RAG no vector_store_id dado.
    """
    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=question,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": 8,
            }],
        )
    except BadRequestError as e:
        print("Erro 400 ao chamar a API:")
        print(e)
        raise
    except Exception as e:
        print("Erro ao chamar a API:")
        print(e)
        raise

    return response.output_text


def chat(vector_store_id: str) -> None:
    """
    Sessão de chat interativa no terminal usando o mesmo vector_store_id,
    com histórico local.
    """
    print(f"Iniciando chat com vector_store_id = {vector_store_id}")
    print("Digite sua pergunta. Comandos: /exit, /quit, /sair, /clear\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo do chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "/sair"):
            print("Saindo do chat.")
            break
        if user_input.lower() == "/clear":
            history.clear()
            print("(Histórico limpo.)")
            continue

        messages = history + [{"role": "user", "content": user_input}]

        try:
            response = client.responses.create(
                model=DEFAULT_MODEL,
                input=messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": 8,
                }],
            )
        except BadRequestError as e:
            print("Erro 400 ao chamar a API:")
            print(e)
            continue
        except Exception as e:
            print("Erro ao chamar a API:")
            print(e)
            continue

        answer = response.output_text
        print(f"\nIA: {answer}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
