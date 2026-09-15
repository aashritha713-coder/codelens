import os
import re
import json
import shutil
import hashlib
import tempfile
import zipfile
import urllib.request
import chromadb
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# Set GitPython quiet env var before importing
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
try:
    import git
    git.Git().version()
    GIT_AVAILABLE = True
except Exception:
    GIT_AVAILABLE = False

# -----------------------------------------------------------------------------
# Streamlit Page & Theme Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeLens — Codebase RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "venv", ".venv",
    "__pycache__", "vendor", ".next", "target", "docs", "site",
    "static", "coverage", ".mypy_cache", ".pytest_cache", "public"
}

SKIP_FILES = {
    "package-lock.json", "poetry.lock", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "composer.lock", "Pipfile.lock"
}

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".go", ".rb", ".php",
    ".yml", ".yaml", ".json"
}

def get_language_by_ext(ext: str) -> str:
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
        ".cpp": "cpp", ".c": "c", ".go": "go", ".rb": "ruby",
        ".php": "php", ".yml": "yaml", ".yaml": "yaml", ".json": "json"
    }
    return lang_map.get(ext.lower(), "text")

# -----------------------------------------------------------------------------
# Cached Resources (Model & Vector DB)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    chroma_dir = os.path.join(os.getcwd(), ".chroma_db_store")
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)

# -----------------------------------------------------------------------------
# Section 4: Fast Code Parsing & Regex Chunking
# -----------------------------------------------------------------------------
def parse_file_chunks(file_path: str, rel_path: str, ext: str) -> list[dict]:
    try:
        if os.path.getsize(file_path) > 200000:
            return []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    if not lines or len(lines) > 2000:
        return []

    total_lines = len(lines)
    chunks = []
    language = get_language_by_ext(ext)

    symbol_patterns = {
        ".py": [
            r"^\s*(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        ],
        ".js": [
            r"\b(function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\b(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>",
            r"\b(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?function"
        ],
        ".ts": [
            r"\b(function|class|interface|type|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\b(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>",
            r"\b(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?function"
        ],
        ".jsx": [
            r"\b(function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\b(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
        ],
        ".tsx": [
            r"\b(function|class|interface|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\b(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
        ],
        ".java": [
            r"\b(?:public|private|protected|static|\s)*\s*(?:class|interface|enum|void|[A-Z][a-zA-Z0-9_<>]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        ],
        ".cpp": [
            r"\b(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{"
        ],
        ".c": [
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{"
        ],
        ".go": [
            r"\bfunc\s+(?:\([^)]+\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\btype\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:struct|interface)"
        ],
        ".rb": [
            r"^\s*(def|class|module)\s+([a-zA-Z_][a-zA-Z0-9_:]*)"
        ],
        ".php": [
            r"\b(function|class|trait|interface)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        ]
    }

    patterns = symbol_patterns.get(ext.lower(), [])
    symbol_matches = []

    for idx, line in enumerate(lines):
        if len(line) > 500:
            continue
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                sym = m.group(2) if len(m.groups()) >= 2 else m.group(1)
                symbol_matches.append((idx, sym))
                break

    if symbol_matches:
        for idx, (start_i, sym_name) in enumerate(symbol_matches):
            next_start_i = symbol_matches[idx + 1][0] if (idx + 1 < len(symbol_matches)) else total_lines
            end_i = max(start_i + 1, next_start_i)
            chunk_text = "".join(lines[start_i:end_i])
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "file_path": rel_path.replace("\\", "/"),
                    "start_line": start_i + 1,
                    "end_line": end_i,
                    "symbol_name": sym_name,
                    "language": language,
                    "source": "repo_code"
                })
    else:
        # Generic Fallback (~40 lines with 5 line overlap)
        chunk_size = 40
        overlap = 5
        start = 0
        while start < total_lines:
            end = min(start + chunk_size, total_lines)
            chunk_text = "".join(lines[start:end])
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "file_path": rel_path.replace("\\", "/"),
                    "start_line": start + 1,
                    "end_line": end,
                    "symbol_name": "block",
                    "language": language,
                    "source": "repo_code"
                })
            if end == total_lines:
                break
            start += (chunk_size - overlap)

    return chunks

def download_github_repo_zip(repo_url: str, target_dir: str) -> tuple[str, str]:
    cleaned_url = repo_url.rstrip("/").removesuffix(".git")
    parts = cleaned_url.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
    owner, repo_name = parts[-2], parts[-1]

    commit_sha = "main-branch"
    zip_path = os.path.join(target_dir, "repo.zip")

    zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/main.zip"
    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(resp, out_file)
    except Exception:
        zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/master.zip"
        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(zip_path, "wb") as out_file:
            shutil.copyfileobj(resp, out_file)
        commit_sha = "master-branch"

    extract_dir = os.path.join(target_dir, "extracted")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    subfolders = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    actual_repo_dir = subfolders[0] if subfolders else extract_dir
    return actual_repo_dir, commit_sha

# -----------------------------------------------------------------------------
# Section 3 & 5: Repo Ingestion & Vector DB Indexing
# -----------------------------------------------------------------------------
def index_repository(repo_url: str, status_box) -> dict:
    temp_dir = tempfile.mkdtemp(prefix="codelens_repo_")
    try:
        if GIT_AVAILABLE:
            status_box.write("⚙️ Step 1/3: Ingesting repository via GitPython (shallow clone)...")
            repo = git.Repo.clone_from(repo_url.strip(), temp_dir, depth=1)
            commit_sha = repo.head.commit.hexsha[:7]
            repo_root = temp_dir
        else:
            status_box.write("⚙️ Step 1/3: Ingesting repository via GitHub Zip Stream...")
            repo_root, commit_sha = download_github_repo_zip(repo_url.strip(), temp_dir)

        status_box.write("⚙️ Step 2/3: Fast-parsing code files into symbol chunks...")
        all_chunks = []
        file_count = 0

        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for file in files:
                if file in SKIP_FILES or file.endswith(".min.js") or file.endswith(".min.css"):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_root)
                    file_chunks = parse_file_chunks(full_path, rel_path, ext)
                    if file_chunks:
                        all_chunks.extend(file_chunks)
                        file_count += 1

        status_box.write(f"⚙️ Step 3/3: Vector embedding {len(all_chunks)} chunks across {file_count} files...")
        embedding_model = get_embedding_model()
        chroma_client = get_chroma_client()

        collection_name = "repo_" + hashlib.md5(repo_url.strip().encode()).hexdigest()

        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if all_chunks:
            documents = [c["text"] for c in all_chunks]
            metadatas = [{
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "symbol_name": c["symbol_name"],
                "language": c["language"],
                "source": c["source"]
            } for c in all_chunks]
            ids = [f"{c['file_path']}:{c['start_line']}-{c['end_line']}:{i}" for i, c in enumerate(all_chunks)]

            embeddings = embedding_model.encode(documents, batch_size=128, show_progress_bar=False).tolist()
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        status_box.write(f"✅ Ready! Indexed {file_count} files ({len(all_chunks)} chunks). Commit SHA: `{commit_sha}`")

        return {
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "collection_name": collection_name,
            "file_count": file_count,
            "chunk_count": len(all_chunks)
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# -----------------------------------------------------------------------------
# Section 6 & 8: Ultra-Fast Hybrid Retrieval Engine
# -----------------------------------------------------------------------------
def retrieve_context(query: str, collection_name: str):
    chroma_client = get_chroma_client()
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    except Exception:
        return [], False, None

    embedding_model = get_embedding_model()
    query_emb = embedding_model.encode(query).tolist()

    # 1. Section 8 Check: Self-Improving Feedback Ledger (human_verified)
    verified_doc = None
    is_verified = False

    try:
        hv_results = collection.query(
            query_embeddings=[query_emb],
            n_results=1,
            where={"source": "human_verified"}
        )
        if hv_results and hv_results.get("documents") and len(hv_results["documents"][0]) > 0:
            distances = hv_results.get("distances", [[1.0]])[0]
            dist = distances[0] if distances else 1.0
            similarity = 1.0 - dist
            if similarity >= 0.70 or dist <= 0.30:
                verified_doc = {
                    "text": hv_results["documents"][0][0],
                    "metadata": hv_results["metadatas"][0][0]
                }
                is_verified = True
    except Exception:
        pass

    # 2. Section 6 Fast Top-15 Vector Retrieval + Symbol Keyword Boosting
    retrieved_chunks = []
    retrieved_ids = set()

    try:
        repo_results = collection.query(
            query_embeddings=[query_emb],
            n_results=15,
            where={"source": "repo_code"}
        )
        if repo_results and repo_results.get("documents"):
            docs = repo_results["documents"][0]
            metas = repo_results["metadatas"][0]
            ids = repo_results["ids"][0]

            query_words = [w.strip().lower() for w in re.split(r'\W+', query) if len(w.strip()) >= 3]

            top_5_docs = docs[:5]
            top_5_metas = metas[:5]
            top_5_ids = ids[:5]

            for d, m, i in zip(top_5_docs, top_5_metas, top_5_ids):
                retrieved_chunks.append({"text": d, "metadata": m})
                retrieved_ids.add(i)

            if len(docs) > 5 and query_words:
                for d, m, i in zip(docs[5:], metas[5:], ids[5:]):
                    if i not in retrieved_ids:
                        sym_name = str(m.get("symbol_name", "")).lower()
                        if any(w in sym_name for w in query_words):
                            retrieved_chunks.append({"text": d, "metadata": m})
                            retrieved_ids.add(i)
                            if len(retrieved_chunks) >= 8:
                                break
    except Exception:
        pass

    return retrieved_chunks, is_verified, verified_doc

# -----------------------------------------------------------------------------
# Section 7 & Section 9: Resilient Groq Model Fallback Chain
# -----------------------------------------------------------------------------
def generate_llm_response(query: str, retrieved_chunks: list, is_diff_mode: bool, verified_doc: dict, api_key: str) -> str:
    if not api_key:
        return "⚠️ Groq API key is missing. Please provide your `GROQ_API_KEY` in the sidebar."

    client = Groq(api_key=api_key)

    code_context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        m = chunk["metadata"]
        code_context_str += f"\n--- SNIPPET {idx} ---\n"
        code_context_str += f"File: {m.get('file_path')}\n"
        code_context_str += f"Symbol: {m.get('symbol_name')}\n"
        code_context_str += f"Lines: {m.get('start_line')} - {m.get('end_line')}\n"
        code_context_str += f"Code:\n{chunk['text']}\n"

    verified_context_str = ""
    if verified_doc:
        verified_context_str = (
            f"\n=== TEAM-VERIFIED HUMAN KNOWLEDGE (HIGH PRIORITY OVER RAW CODE) ===\n"
            f"{verified_doc['text']}\n"
            f"=================================================================\n"
        )

    if is_diff_mode:
        system_prompt = (
            "You are CodeLens, an expert software developer.\n"
            "The user is requesting a code change or refactoring proposal for the retrieved code.\n"
            "Instructions:\n"
            "1. Provide a short, direct rationale for the change.\n"
            "2. Provide the proposed change as a unified code diff using standard `+` (lines added) and `-` (lines removed) prefixes against the original code snippets.\n"
            "3. State the file name and symbol/function name being modified.\n"
            "4. NEVER claim the code has been updated in the repository — this is a suggestion surface only.\n"
        )
        user_prompt = f"Change Request: {query}\n{verified_context_str}\nRetrieved Codebase Context:\n{code_context_str}"
    else:
        system_prompt = (
            "You are CodeLens, an expert codebase RAG assistant.\n"
            "Answer the user's question accurately using ONLY the provided codebase context and team-verified knowledge.\n"
            "Instructions:\n"
            "1. ALWAYS cite the exact File Name, Function/Class Name, and Line Range (e.g. `src/auth.py` -> `validate_password` [L15-L42]) for every technical explanation.\n"
            "2. If the provided code snippets do NOT answer the question, explicitly state: 'I don't see this in the retrieved code'. Never invent or assume unlisted code.\n"
            "3. Maintain a concise, technical, developer-friendly tone.\n"
        )
        user_prompt = f"Question: {query}\n{verified_context_str}\nRetrieved Codebase Context:\n{code_context_str}"

    # Updated fallback list of active Groq model identifiers
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]

    last_err = None
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_name,
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            # Bypasses decommissioned, 404, 400, or not found models to try next model in chain
            if any(k in err_msg for k in ["decommissioned", "not_found", "not found", "does not exist", "404", "400", "invalid_request_error"]):
                continue
            else:
                break

    return f"❌ Error communicating with Groq API: {str(last_err)}"

# -----------------------------------------------------------------------------
# Unified Verified Memory Store (Section 8 & Section 9)
# -----------------------------------------------------------------------------
def save_human_verified_memory(collection_name: str, query: str, content: str, doc_type: str = "correction") -> bool:
    chroma_client = get_chroma_client()
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        embedding_model = get_embedding_model()

        doc_text = f"Question/Request: {query}\nVerified Content ({doc_type}):\n{content}"
        doc_emb = embedding_model.encode(doc_text).tolist()

        doc_id = f"verified_{hashlib.md5(doc_text.encode()).hexdigest()[:10]}"
        metadata = {
            "source": "human_verified",
            "linked_question": query,
            "doc_type": doc_type,
            "symbol_name": "human_feedback",
            "file_path": "memory/feedback_ledger",
            "start_line": 1,
            "end_line": 1,
            "language": "markdown"
        }

        collection.add(
            embeddings=[doc_emb],
            documents=[doc_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return True
    except Exception as e:
        st.error(f"Failed to save correction to vector DB: {e}")
        return False

# -----------------------------------------------------------------------------
# Streamlit Application State & Main UI
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None

# Sidebar Configuration
st.sidebar.title("⚙️ CodeLens Config")

groq_api_key = os.environ.get("GROQ_API_KEY", "")
user_key_input = st.sidebar.text_input("Groq API Key", value=groq_api_key, type="password", help="Enter your GROQ_API_KEY")
if user_key_input:
    groq_api_key = user_key_input

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Indexed Repository Status")
if st.session_state.indexed_repo:
    info = st.session_state.indexed_repo
    st.sidebar.markdown(f"**URL:** `{info['repo_url']}`")
    st.sidebar.markdown(f"**Commit SHA:** `{info['commit_sha']}`")
    st.sidebar.markdown(f"**Files Parsed:** `{info['file_count']}`")
    st.sidebar.markdown(f"**Code Chunks:** `{info['chunk_count']}`")
    
    if st.sidebar.button("🗑️ Clear Indexed Repository"):
        st.session_state.indexed_repo = None
        st.session_state.chat_history = []
        st.rerun()
else:
    st.sidebar.info("No repository indexed yet.")

# Main Application Layout
st.title("CodeLens")
st.caption("Ask any question about an unfamiliar codebase with exact citations & self-improving team memory.")

st.markdown("---")

# Section 3: Ingestion Form
st.subheader("1️⃣ Index Repository")
repo_col1, repo_col2 = st.columns([4, 1])

default_url = st.session_state.indexed_repo["repo_url"] if st.session_state.indexed_repo else "https://github.com/pallets/flask"
repo_url_input = repo_col1.text_input("Public GitHub Repository URL", value=default_url, placeholder="https://github.com/user/repository")
index_button = repo_col2.button("🚀 Index Repository", use_container_width=True)

if index_button:
    if not repo_url_input.strip():
        st.error("Please enter a valid GitHub repository URL.")
    else:
        with st.status("Indexing repository...", expanded=True) as status_box:
            try:
                result = index_repository(repo_url_input, status_box)
                st.session_state.indexed_repo = result
                st.session_state.chat_history = []
                st.success("Indexing complete! You can now ask questions below.")
            except Exception as e:
                status_box.update(label=f"Failed to index repository: {e}", state="error")
                st.error(f"Error during ingestion: {str(e)}")

st.markdown("---")

# Q&A / Chat History Section
st.subheader("2️⃣ Codebase Exploration & Q&A")

if not st.session_state.indexed_repo:
    st.info("👈 Please enter a public GitHub URL and click **Index Repository** to get started.")
else:
    # Display Scrolling Chat History (Last 5 Turns)
    for idx, turn in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"### ❓ **Question:** {turn['question']}")
            
            # Badge Rendering (Section 8)
            if turn.get("is_verified"):
                st.success("✅ Team-verified")
            else:
                st.warning("⚠️ AI-generated, unreviewed")

            if turn.get("is_diff_mode"):
                st.info("🛠️ **Suggested Code Change** *(AI-suggested, unreviewed — not applied to your repo)*")

            # Answer Content
            st.markdown(turn["answer"])

            # Retrieved Code Chunks (Section 7 Citation & Expanders)
            if turn.get("chunks"):
                with st.expander(f"📚 View {len(turn['chunks'])} Referenced Code Snippets"):
                    for c_idx, chunk in enumerate(turn["chunks"], 1):
                        m = chunk["metadata"]
                        st.caption(f"Snippet {c_idx}: **{m.get('file_path')}** | Symbol: `{m.get('symbol_name')}` | Lines: {m.get('start_line')}-{m.get('end_line')}")
                        st.code(chunk["text"], language=m.get("language", "python"))

            # Section 8 & 9 Feedback & Adoption Controls
            col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 4])
            
            thumbs_up = col_fb1.button("👍 Useful", key=f"tu_{idx}")
            thumbs_down = col_fb2.button("👎 Poor", key=f"td_{idx}")
            
            if thumbs_up:
                st.toast("Thank you for your feedback! 👍")
            if thumbs_down:
                st.toast("Feedback recorded. 👎")

            # Correction Box (Section 8 Differentiator A)
            correct_key = f"show_correct_{idx}"
            if correct_key not in st.session_state:
                st.session_state[correct_key] = False

            if col_fb3.button("📝 Correct this explanation", key=f"btn_corr_{idx}"):
                st.session_state[correct_key] = not st.session_state[correct_key]

            if st.session_state[correct_key]:
                with st.form(key=f"form_corr_{idx}"):
                    corr_text = st.text_area("Enter senior engineer correction:", placeholder="Explain the correct behavior or authentication handling...")
                    submit_corr = st.form_submit_button("Save to Team Memory")
                    if submit_corr:
                        if corr_text.strip():
                            col_name = st.session_state.indexed_repo["collection_name"]
                            if save_human_verified_memory(col_name, turn["question"], corr_text.strip(), doc_type="correction"):
                                st.success("✅ Correction saved to self-improving memory! Ask again to see verified response.")
                                st.session_state[correct_key] = False
                        else:
                            st.warning("Please enter correction text.")

            # Adopt as Canonical (Section 9 Differentiator B)
            if turn.get("is_diff_mode"):
                if st.button("⭐ Adopt as canonical solution", key=f"btn_adopt_{idx}"):
                    col_name = st.session_state.indexed_repo["collection_name"]
                    if save_human_verified_memory(col_name, turn["question"], turn["answer"], doc_type="canonical_diff"):
                        st.success("✅ Adopted as canonical solution! Saved to team vector memory.")

            st.markdown("---")

    # Bottom Query Form (Section 9 Toggle + Input)
    with st.form(key="query_form", clear_on_submit=True):
        col_inp, col_tgl = st.columns([5, 1.5])
        user_query = col_inp.text_input("Ask a question about the code or request a refactor:", placeholder="e.g. Where is password validation handled?")
        suggest_change = col_tgl.checkbox("Suggest a change", help="Toggle to request code changes / unified diffs")
        submit_query = st.form_submit_button("Submit Question / Request", use_container_width=True)

    if submit_query:
        if not user_query.strip():
            st.warning("Please enter a question or change request.")
        elif not groq_api_key:
            st.error("Please enter your `GROQ_API_KEY` in the sidebar.")
        else:
            collection_name = st.session_state.indexed_repo["collection_name"]
            with st.spinner("Retrieving codebase context and generating answer..."):
                retrieved_chunks, is_verified, verified_doc = retrieve_context(user_query.strip(), collection_name)
                answer = generate_llm_response(
                    query=user_query.strip(),
                    retrieved_chunks=retrieved_chunks,
                    is_diff_mode=suggest_change,
                    verified_doc=verified_doc,
                    api_key=groq_api_key
                )

                new_turn = {
                    "question": user_query.strip(),
                    "answer": answer,
                    "is_diff_mode": suggest_change,
                    "chunks": retrieved_chunks,
                    "is_verified": is_verified
                }

                st.session_state.chat_history.append(new_turn)
                if len(st.session_state.chat_history) > 5:
                    st.session_state.chat_history.pop(0)

                st.rerun()
