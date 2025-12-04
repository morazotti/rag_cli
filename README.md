# rag_cli.py – Minimal RAG CLI com OpenAI 🚀

Autor: Nícolas Morazotti  

---

## Visão geral 🧠

Este repositório contém um único script `rag_cli.py`, um **CLI minimalista de RAG** (Retrieval-Augmented Generation) usando:

- OpenAI Vector Stores 📦  
- Ferramenta `file_search` da API de Responses (`client.responses.create`) 🔍  
- Suporte a múltiplas extensões de arquivos (incluindo conversão automática de `.org` → `.md`) ✨

O objetivo é ter um utilitário simples de terminal para:

- criar índices de documentos locais em um vector store; 🗂️
- fazer perguntas pontuais com RAG; ❓
- manter sessões de chat em cima dos seus arquivos; 💬
- estender um índice existente com novos arquivos. ➕

Foi desenvolvido de forma experimental / exploratória, em estilo **“vibe‑coding”**: iterando rápido no código diretamente, ajustando a experiência de uso ao longo do caminho. 🎛️

---

## Requisitos ⚙️

- Python 3.8+ 🐍  
- Biblioteca OpenAI:

  ```sh
  pip install "openai>=1.90.0"
  ```

- Para arquivos `.org` (opcional, mas recomendado):
  - `pandoc` instalado no sistema (para converter `.org` em `.md` temporário) 🔁

---

## Configuração da API Key 🔑

O script procura a chave da API da OpenAI de duas formas:

1. **Variável de ambiente** 🌎:

   ```sh
   export OPENAI_API_KEY="sk-..."
   ```

2. **Arquivo `~/.authinfo`** (estilo Emacs/Tramp) 📄, com uma linha no formato:

   ```text
   machine api.openai.com login apikey password sk-...
   ```

Se nenhuma chave for encontrada, o script aborta informando como configurar. ⚠️

---

## Funcionalidades principais 🧰

Comando base:

```sh
python rag_cli.py <comando> [ARGS...]
```

Comandos disponíveis:

- `index PATH_OR_GLOB` 📥
- `ask VECTOR_STORE_ID|auto|PATH_OR_GLOB "pergunta"` ❓
- `chat auto|PATH_OR_GLOB|VECTOR_STORE_ID` 💬
- `extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB` ➕
- `list` 📃

---

### 1. `index` – criar um novo índice (vector store) 🗂️

Cria um vector store na OpenAI a partir de um diretório ou glob local.

- Expande `PATH_OR_GLOB` (`~`, `$HOME`, `**`, etc.). 🔍  
- Recorre diretórios quando for um path. 📁  
- Filtra por extensões suportadas:
  - `.pdf`, `.txt`, `.md`, `.rtf`
  - `.docx`, `.pptx`
  - `.csv`, `.tsv`
  - `.html`, `.htm`
  - `.json`, `.xml`
  - `.org` (convertido automaticamente para `.md` via `pandoc`) 📝

Antes de enviar os arquivos, o script:

- calcula uma **estimativa grosseira de custo** (chars → tokens ≈ chars/4); 📊  
- mostra o custo estimado usando um preço de referência (`text-embedding-3-small`); 💵  
- pede confirmação interativa (`[y/N]`). ✅❌

Se o usuário confirmar:

- cria um vector store (`client.vector_stores.create`); 🏗️  
- faz upload de cada arquivo (`client.vector_stores.files.upload_and_poll`); ⬆️  
- converte `.org` para `.md` temporário com `pandoc`; 🔁  
- salva o `vector_store_id` em um cache local (`$HOME/.rag_vector_stores.json`). 💾

**Uso:**

```sh
python rag_cli.py index PATH_OR_GLOB
```

**Exemplos:**

```sh
# Indexar um diretório inteiro de notas
python rag_cli.py index "$HOME/notes"

# Indexar apenas arquivos org em subdiretórios
python rag_cli.py index "$HOME/notes/**/*.org"
```

O cache associa **chaves canônicas** (diretório absoluto ou glob expandido) ao `vector_store_id`, e também guarda o último id usado (`_last`). 📚

---

### 2. `ask` – pergunta única com RAG ❓

Faz uma pergunta pontual usando RAG em cima de um vector store.

Chama `client.responses.create` com:

- `model="gpt-4.1-mini"` 🤖  
- `tools=[{"type": "file_search", "vector_store_ids": [...]}]` 🔎

Não mantém histórico de conversa; é uma interação independente. ⚡

Formas de uso:

1. **Passando um `vector_store_id` explícito (`vs_...`)** 🆔:

   ```sh
   python rag_cli.py ask vs_XXXXXXXX "Qual é o objetivo do projeto?"
   ```

2. **Usando o último índice criado (`auto`)** 🔁:

   ```sh
   python rag_cli.py ask auto "Resuma as principais ideias dos arquivos."
   ```

3. **Usando um `PATH_OR_GLOB` que já foi indexado**  
   (o script resolve esta chave via cache em `$HOME/.rag_vector_stores.json`) 📂:

   ```sh
   python rag_cli.py ask "$HOME/notes" "Quais tarefas estão pendentes?"
   ```

---

### 3. `chat` – sessão interativa estilo ChatGPT 💬

Abre um loop de REPL no terminal, mantendo um **histórico local** de mensagens (user/assistant) e sempre usando o mesmo vector store.

Funcionalidades:

- histórico em memória; 🧾  
- comandos especiais:
  - `/exit`, `/quit`, `/sair` → encerra o chat; ❌
  - `/clear` → limpa o histórico local; 🧹
- integração com RAG em cada pergunta:

  ```python
  client.responses.create(
      model="gpt-4.1-mini",
      input=messages,          # histórico + nova pergunta
      tools=[{
          "type": "file_search",
          "vector_store_ids": [vector_store_id],
          "max_num_results": 8,
      }],
  )
  ```

**Formas de uso:**

```sh
# Chat usando o último índice criado
python rag_cli.py chat auto

# Chat usando uma sessão indexada específica
python rag_cli.py chat "$HOME/notes"

# Chat passando um vector_store_id diretamente
python rag_cli.py chat vs_XXXXXXXX
```

---

### 4. `extend` – anexar novos arquivos a um índice existente ➕

Permite **estender** um vector store já existente com novos arquivos, sem criar outro índice e sem perder o que já existe.

Fluxo:

1. Recebe: 📥  
   - `PATH_OR_GLOB_EXISTENTE` → identifica qual sessão/índice usar;  
   - `NOVOS_ARQUIVOS_GLOB` → arquivos adicionais a enviar.  
2. Resolve o `vector_store_id` via cache a partir de `PATH_OR_GLOB_EXISTENTE`. 🗃️  
3. Aplica o mesmo fluxo de filtragem/estimativa de custo/upload usado em `index`. 📊⬆️  
4. Atualiza o cache marcando este `vector_store_id` como o último utilizado. 🔄

**Uso:**

```sh
python rag_cli.py extend PATH_OR_GLOB_EXISTENTE NOVOS_ARQUIVOS_GLOB
```

**Exemplo:**

```sh
python rag_cli.py extend "$HOME/notes" "$HOME/notes/inbox/**/*.org"
```

---

### 5. `list` – listar vector stores cacheados 📃

Mostra o conteúdo do arquivo de cache:

- caminho do cache (`$HOME/.rag_vector_stores.json`); 📍  
- chaves de sessão (diretório absoluto ou glob) → `vector_store_id`; 🔗  
- destaca qual foi o último índice usado (`_last`). ⭐

**Uso:**

```sh
python rag_cli.py list
```

---

## Detalhes de implementação 🧬

### Cache de vector stores 💾

O cache é um JSON em:

- `VECTOR_STORE_CACHE_FILE = ~/.rag_vector_stores.json`

Estrutura aproximada (formato simplificado):

```json
{
  "/abs/path/to/notes": "vs_abc123",
  "~/notes/**/*.org":  "vs_def456",
  "_last":             "vs_def456"
}
```

Funções principais:

- `canonical_key(path_or_glob)` → normaliza a chave de cache; 🧩  
- `save_vector_store_id_for_key(key, vs_id)`; 💾  
- `load_vector_store_id_for_key(key)`; 📥  
- `load_last_vector_store_id()`. 🔙

> Obs.: No código atual, o formato real é um pouco mais estruturado (`sessions`, `files_per_vs`, `_last`), mas o conceito geral acima permanece. 🧱

---

### Estimativa de custo 💵

Para todos os arquivos que serão enviados, o script:

1. lê os arquivos em texto (ignorando erros de encoding); 📖  
2. soma `len(text)` para obter o total de caracteres; ➕  
3. estima tokens como `chars / 4`; 📏  
4. calcula custo aproximado com:

   - `EMBED_PRICE_PER_MILLION = 0.02` (USD / 1M tokens); 💲

5. pergunta ao usuário se deve prosseguir. ✅❓

Essa estimativa é **grosseira** e serve apenas para evitar surpresas. 🎯

---

### Conversão automática de `.org` para `.md` 🔁

Arquivos `.org` são tratados via:

- `convert_org_to_md_temp(org_path)`:
  - cria um diretório temporário (`tempfile.mkdtemp`); 🗃️  
  - chama `pandoc org_path -o md_path`; 🧰  
  - retorna o `.md` resultante; 📄  
- após o upload, o diretório temporário é removido (`shutil.rmtree`). 🧹

Caso `pandoc` não esteja instalado, o script falha com uma mensagem clara sobre a necessidade de instalação para lidar com `.org`. ⚠️

---

### Manipulação de erros 🧯

- `BadRequestError` (400) da API é capturado para:
  - uploads de arquivos (`upload_and_poll`); ⬆️  
  - chamadas de `responses.create`. 📡  
- Outros erros são capturados genericamente e impressos no terminal. 🖥️  
- No caso de `index`/`extend`, o script tenta continuar com os demais arquivos, mesmo que um falhe. 🔁

---

## Motivação e estilo “vibe‑coding” 🎧

Este projeto nasceu de uma necessidade prática: ter um CLI simples para explorar RAG em cima de notas e arquivos locais, sem precisar de um framework pesado. 🪶

Ele foi desenvolvido em modo **“vibe‑coding”**:

- iterando o design diretamente no código; 🧪  
- testando no REPL/terminal e ajustando a ergonomia dos comandos; ⌨️  
- priorizando fluxo rápido de experimentação em vez de arquitetura complexa. ⚡

Por isso, o script é:

- **monolítico**, mas fácil de ler; 📜  
- focado em **casos reais de uso** (notas pessoais, pastas de estudos etc.); 🎯  
- uma boa base para forks e customizações locais. 🌱

---

## Como começar 🚀

1. **Instale dependências:** ⚙️

   ```sh
   pip install "openai>=1.90.0"
   # pandoc opcional, mas recomendado:
   # sudo apt install pandoc
   ```

2. **Configure sua chave de API:** 🔑

   - via `OPENAI_API_KEY`, ou  
   - via `~/.authinfo`.

3. **Crie um índice:** 🗂️

   ```sh
   python rag_cli.py index "$HOME/notes"
   ```

4. **Faça perguntas:** ❓

   ```sh
   python rag_cli.py ask auto "Resuma as ideias principais das notas."
   ```

5. **Abra um chat interativo:** 💬

   ```sh
   python rag_cli.py chat auto
   ```

Sinta‑se à vontade para adaptar o script ao seu fluxo de trabalho, trocar o modelo, ajustar limites ou integrar com outras ferramentas em cima do vector store. 🛠️
