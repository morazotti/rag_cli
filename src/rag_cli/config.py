from pathlib import Path

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

# Modelo de chat/RAG padrão
DEFAULT_MODEL = "gpt-4.1-mini"
