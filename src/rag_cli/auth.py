import os
import re
from pathlib import Path


def parse_authinfo() -> str | None:
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


def load_api_key() -> str:
    """
    Prioriza variável de ambiente OPENAI_API_KEY.
    Caso contrário, tenta ~/.authinfo.
    """
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    key = parse_authinfo()
    if key:
        return key

    raise RuntimeError(
        "OpenAI API key não encontrada.\n"
        "Defina OPENAI_API_KEY no ambiente ou adicione ao ~/.authinfo:\n"
        "  machine api.openai.com login apikey password sk-..."
    )
