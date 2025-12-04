#!/usr/bin/env python3
"""
Minimal RAG CLI usando OpenAI vector stores + Responses + file_search.

Comandos principais:

  python rag_cli.py index PATH_OR_GLOB
      -> cria um vector store novo e indexa arquivos (md, txt, org, etc.)

  python rag_cli.py extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB
      -> reusa o MESMO vector store daquela sessão e
         INDEXA APENAS ARQUIVOS NOVOS (não reindexa repetidos)

  python rag_cli.py ask auto "sua pergunta"
  python rag_cli.py ask PATH_OR_GLOB "sua pergunta"
  python rag_cli.py ask vs_abc123... "sua pergunta"

  python rag_cli.py chat auto
  python rag_cli.py chat PATH_OR_GLOB
  python rag_cli.py chat vs_abc123...

  python rag_cli.py list
      -> lista sessões (PATH_OR_GLOB) e seus vector_store_ids

Requer:
    pip install "openai>=1.90.0"
e, para arquivos .org:
    pandoc instalado no sistema.
"""

import os
import sys
import re
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from glob import glob

from openai import OpenAI
try:
    from openai import BadRequestError
except Exception:  # fallback, caso não exista essa classe
    BadRequestError = Exception


# ---------- API key: env ou ~/.authinfo ----------

def parse_authinfo():
    """
    Procura a API key no arquivo ~/.authinfo no formato:

      machine api.openai.com login apikey password sk-...

    Retorna a key ou None.
    """
    authinfo = Path.home() / ".authinfo"
    if not authinfo.exists():
        return None

    text = authinfo.read_text()
    pattern = re.compile(r"machine\s+api\.openai\.com.*password\s+(\S+)")
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            return m.group(1)
    return None


def load_api_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    key = parse_authinfo()
    if key:
        return key

    raise RuntimeError(
        "OpenAI API key não encontrada.\n"
        "Defina a variável de ambiente OPENAI_API_KEY ou adicione uma linha ao ~/.authinfo:\n"
        "  machine api.openai.com login apikey password sk-..."
    )


client = OpenAI(api_key=load_api_key())


# ---------- Cache de vector stores (por diretório/glob) + arquivos indexados ----------

VECTOR_STORE_CACHE_FILE = os.path.expanduser("~/.rag_vector_stores.json")


def _empty_cache():
    return {
        "sessions": {},       # key -> vs_id
        "files_per_vs": {},   # vs_id -> [abs_path, ...]
        "_last": None         # último vs_id usado
    }


def _load_cache():
    path = VECTOR_STORE_CACHE_FILE
    if not os.path.exists(path):
        return _empty_cache()

    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return _empty_cache()

    # Se já tem formato novo (sessions/files_per_vs), só completa defaults
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


def _save_cache(cache):
    # Garante chaves padrão
    base = _empty_cache()
    base.update(cache or {})
    with open(VECTOR_STORE_CACHE_FILE, "w") as f:
        json.dump(base, f, indent=2)


def canonical_key(path_or_glob):
    """
    Normaliza um diretório ou glob para ser usado como chave de cache.
    - Diretórios -> caminho absoluto
    - Globs/padrões -> string expandida (~, $HOME, etc), como está
    """
    expanded = os.path.expanduser(os.path.expandvars(path_or_glob))
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)
    else:
        return expanded


def save_vector_store_id_for_key(key, vs_id):
    cache = _load_cache()
    cache["sessions"][key] = vs_id
    cache["_last"] = vs_id
    _save_cache(cache)


def load_vector_store_id_for_key(key):
    cache = _load_cache()
    return cache["sessions"].get(key)


def load_last_vector_store_id():
    cache = _load_cache()
    return cache.get("_last")


def add_indexed_files(vs_id, files):
    """
    Adiciona arquivos (paths absolutos) como já indexados para um vs_id.
    Não duplica paths; mantém lista ordenada.
    """
    cache = _load_cache()
    fps = cache.setdefault("files_per_vs", {})
    existing = set(fps.get(vs_id, []))
    new_abs = {os.path.abspath(p) for p in files}
    combined = sorted(existing.union(new_abs))
    fps[vs_id] = combined
    _save_cache(cache)


def get_indexed_files(vs_id):
    """
    Retorna um set de paths absolutos já indexados para esse vs_id.
    """
    cache = _load_cache()
    fps = cache.get("files_per_vs", {})
    return set(fps.get(vs_id, []))


def list_vector_stores():
    cache = _load_cache()
    sessions = cache.get("sessions", {})
    if not sessions:
        print("No cached vector stores.")
        return

    last_vs = cache.get("_last")

    print(f"Cache file: {VECTOR_STORE_CACHE_FILE}\n")
    print("Cached vector stores (per directory/glob key):")
    for key, vs_id in sessions.items():
        marker = "  (last used)" if vs_id == last_vs else ""
        print(f"- {key} -> {vs_id}{marker}")


def resolve_vector_store_id(arg):
    """
    Resolve um argumento (auto, vs_..., path/glob) para um vector_store_id real.
    """
    # auto -> último vs_id
    if arg == "auto":
        vs = load_last_vector_store_id()
        if not vs:
            raise RuntimeError(
                "Nenhum vector store em cache.\n"
                "Rode antes:\n"
                "  python rag_cli.py index PATH_OR_GLOB"
            )
        return vs

    # ID explícito
    if arg.startswith("vs_"):
        return arg

    # path/glob -> chave de cache
    key = canonical_key(arg)
    vs = load_vector_store_id_for_key(key)
    if not vs:
        raise RuntimeError(
            "Nenhum vector store em cache para essa chave:\n"
            f"  {key}\n"
            "Faça o índice antes com:\n"
            f"  python rag_cli.py index \"{arg}\""
        )
    return vs


# ---------- Tipos de arquivo suportados ----------

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".rtf",
    ".docx", ".pptx",
    ".csv", ".tsv",
    ".html", ".htm",
    ".json", ".xml",
    ".org",  # tratado via conversão automática para .md
}


# ---------- Estimativa de custo ----------

EMBED_PRICE_PER_MILLION = 0.02  # USD por 1M tokens (aprox text-embedding-3-small)


def estimate_cost_for_files(file_paths):
    total_chars = 0

    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                total_chars += len(f.read())
        except Exception as e:
            print(f"  [warn] Não consegui ler {path} para estimativa: {e}")

    est_tokens = total_chars / 4.0
    est_cost = (est_tokens / 1_000_000.0) * EMBED_PRICE_PER_MILLION
    return int(total_chars), int(est_tokens), est_cost


def show_cost_and_confirm(file_paths):
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
        sys.exit(0)


# ---------- Conversão .org -> .md temporário ----------

def convert_org_to_md_temp(org_path):
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


def prepare_file_for_upload(path):
    """
    Prepara um arquivo para upload.
    - Se for .org -> converte para .md temporário e retorna (upload_path, tmp_path)
    - Caso contrário -> (path, None)
    """
    if path.lower().endswith(".org"):
        tmp_md = convert_org_to_md_temp(path)
        return tmp_md, tmp_md
    else:
        return path, None


# ---------- Indexação: PATH ou GLOB (cria novo vector store) ----------

def index_path_or_glob(path_or_glob):
    """
    Cria um vector store e envia:
      - todos os arquivos suportados num diretório, OU
      - todos os arquivos suportados que batam com um glob (incluindo **).

    Retorna o vector_store_id.
    """
    expanded = os.path.expanduser(os.path.expandvars(path_or_glob))

    if os.path.isdir(expanded):
        search_pattern = os.path.join(os.path.abspath(expanded), "**", "*")
        paths = glob(search_pattern, recursive=True)
    else:
        paths = glob(expanded, recursive=True)

    file_paths = []
    skipped = []
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

    indexed_success = []

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
                try:
                    shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)
                except Exception:
                    pass

    if indexed_success:
        add_indexed_files(vector_store_id, indexed_success)

    print("Indexação concluída.")
    return vector_store_id


# ---------- Extensão: anexar arquivos novos a um vector store existente ----------

def extend_path_or_glob(vector_store_id, path_or_glob):
    """
    Anexa arquivos novos (ou não) a um vector_store já existente,
    usando um PATH_OR_GLOB arbitrário.

    NÃO reindexa arquivos já enviados para esse vector_store_id
    (usa paths absolutos).
    """
    expanded = os.path.expanduser(os.path.expandvars(path_or_glob))

    if os.path.isdir(expanded):
        search_pattern = os.path.join(os.path.abspath(expanded), "**", "*")
        paths = glob(search_pattern, recursive=True)
    else:
        paths = glob(expanded, recursive=True)

    file_paths = []
    skipped = []
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

    # filtra só arquivos realmente novos para esse vs_id
    already = get_indexed_files(vector_store_id)
    new_files = [p for p in file_paths if p not in already]

    if not new_files:
        print("Nenhum arquivo novo para indexar; todos já estão neste vector store.")
        return

    print(f"Encontrados {len(new_files)} arquivos NOVOS para anexar.")
    show_cost_and_confirm(new_files)

    print(f"\nAnexando arquivos ao vector store existente: {vector_store_id}")
    indexed_success = []

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
                try:
                    shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)
                except Exception:
                    pass

    if indexed_success:
        add_indexed_files(vector_store_id, indexed_success)

    print("Extensão concluída.")


# ---------- Pergunta única (ask) ----------

def ask(vector_store_id, question):
    """
    Pergunta única usando RAG no vector_store_id dado.
    Não mantém histórico (uma chamada independente).
    """
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
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


# ---------- Chat interativo ----------

def chat(vector_store_id):
    """
    Sessão de chat interativa no terminal usando o mesmo vector_store_id,
    com histórico local de conversa (estilo ChatGPT).
    """
    print(f"Iniciando chat com vector_store_id = {vector_store_id}")
    print("Digite sua pergunta. Comandos: /exit, /quit, /sair, /clear\n")

    history = []

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
                model="gpt-4.1-mini",
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


# ---------- CLI ----------

def main(argv):
    if len(argv) < 2:
        print("Uso:")
        print("  python rag_cli.py index PATH_OR_GLOB")
        print("  python rag_cli.py extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB")
        print("  python rag_cli.py ask VECTOR_STORE_ID \"sua pergunta\"")
        print("  python rag_cli.py ask auto \"sua pergunta\"          # último índice")
        print("  python rag_cli.py ask PATH_OR_GLOB \"sua pergunta\"  # índice por diretório/glob")
        print("  python rag_cli.py chat auto                        # chat interativo com último índice")
        print("  python rag_cli.py chat PATH_OR_GLOB                # chat interativo com índice daquele path")
        print("  python rag_cli.py list                             # lista índices cacheados")
        sys.exit(1)

    cmd = argv[1]

    if cmd == "index":
        if len(argv) != 3:
            print("Uso: python rag_cli.py index PATH_OR_GLOB")
            sys.exit(1)
        pattern = argv[2]
        vs_id = index_path_or_glob(pattern)
        key = canonical_key(pattern)
        save_vector_store_id_for_key(key, vs_id)
        print("\n=== VECTOR_STORE_ID (cacheado) ===")
        print(f"Key: {key}")
        print(f"ID : {vs_id}")
        print(f"Salvo em: {VECTOR_STORE_CACHE_FILE}")

    elif cmd == "extend":
        if len(argv) != 4:
            print("Uso:")
            print("  python rag_cli.py extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB")
            print()
            print("Exemplo:")
            print("  python rag_cli.py extend \"$HOME/mdroam\" \"$HOME/mdroam/novas/**/*.org\"")
            sys.exit(1)

        session_key = argv[2]
        new_pattern = argv[3]

        vector_store_id = resolve_vector_store_id(session_key)
        extend_path_or_glob(vector_store_id, new_pattern)

        # marca esse vs como último usado
        key = canonical_key(session_key)
        save_vector_store_id_for_key(key, vector_store_id)

    elif cmd == "ask":
        if len(argv) < 4:
            print("Uso: python rag_cli.py ask VECTOR_STORE_ID \"sua pergunta\"")
            print("     python rag_cli.py ask auto \"sua pergunta\"")
            print("     python rag_cli.py ask PATH_OR_GLOB \"sua pergunta\"")
            sys.exit(1)

        arg2 = argv[2]
        vector_store_id = resolve_vector_store_id(arg2)
        question = " ".join(argv[3:])
        answer = ask(vector_store_id, question)
        print(answer)

    elif cmd == "chat":
        if len(argv) < 3:
            print("Uso: python rag_cli.py chat auto")
            print("     python rag_cli.py chat PATH_OR_GLOB")
            print("     python rag_cli.py chat VECTOR_STORE_ID")
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


if __name__ == "__main__":
    main(sys.argv)
