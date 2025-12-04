from pathlib import Path
import os

# Onde o cache JSON das sessões e arquivos fica salvo
VECTOR_STORE_CACHE_PATH = Path.home() / ".rag_vector_stores.json"

# Extensões de arquivo suportadas para retrieval (nota: .org é convertido p/ .md)
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".rtf",
    ".docx", ".pptx",
    ".csv", ".tsv",
    ".html", ".htm",
    ".json", ".xml",
    ".org",
}

# Preço aproximado do embedding (USD por 1M tokens)
EMBED_PRICE_PER_MILLION = 0.02  # text-embedding-3-small (aprox.)

# Modelo de chat/RAG padrão (pode ser sobrescrito por env)
DEFAULT_MODEL = os.getenv("RAG_CLI_MODEL", "gpt-4.1-mini")

# Prompt inicial / system prompt padrão (opcional; também via env)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "RAG_CLI_SYSTEM",
    ""  # se quiser, você já pode colocar um default aqui
)
