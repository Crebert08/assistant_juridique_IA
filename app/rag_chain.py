import anthropic
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from . import jev

load_dotenv()

PDF_PATH = "Burundi_Code_2017_penal.pdf"
PERSIST_DIR = "rag"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"
MAX_TOKENS = 16000

IN_SCOPE_THRESHOLD = 0.5
GROUNDED_THRESHOLD = 0.7

REFUSAL_ANSWER = (
    "Je ne peux pas répondre à cette question. Consultez un professionnel du droit."
)

SYSTEM_PROMPT = (
    "Vous êtes un assistant juridique spécialisé dans le Code pénal du Burundi. "
    "Votre rôle est de fournir des informations précises et contextuelles sur le droit pénal burundais. "
    "RÈGLES DE RÉPONSE : "
    "• Utilisez UNIQUEMENT les extraits de contexte fournis pour formuler votre réponse "
    "• Les extraits sont fournis dans la balise <extraits>, chacun précédé de son numéro de page "
    "• Citez toujours les articles spécifiques quand ils sont mentionnés dans le contexte "
    "• Si l'information n'est pas présente dans le contexte, indiquez clairement 'Je ne trouve pas cette information dans le contexte fourni' "
    "• Répondez de manière claire et précise en maximum 4 phrases "
    "• Utilisez la terminologie juridique française appropriée "
    "STRUCTURE DE RÉPONSE : "
    "1. Réponse directe avec citation d'article si disponible "
    "2. Explication du contexte juridique basée sur les extraits "
    "3. Mention des sanctions ou implications pratiques si présentes dans le contexte "
    "PRÉCISIONS IMPORTANTES : "
    "• Distinguez entre crimes, délits et contraventions selon les classifications du contexte "
    "• Mentionnez les peines d'emprisonnement, amendes et autres sanctions telles qu'indiquées "
    "• Indiquez les relations entre différentes dispositions quand elles apparaissent dans le contexte "
    "• Maintenez un ton professionnel mais accessible "
    "LIMITATIONS : "
    "• Vous fournissez des informations juridiques, pas des conseils juridiques "
    "• Recommandez la consultation d'un professionnel du droit pour les cas spécifiques "
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
if not vectorstore.get(limit=1)["ids"]:
    data = PyPDFLoader(PDF_PATH).load()  # un Document par page
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    vectorstore.add_documents(text_splitter.split_documents(data))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

_claude = None


def get_claude():
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY
    return _claude


def format_context(docs) -> str:
    extraits = "\n\n".join(
        f"[page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )
    return f"<extraits>\n{extraits}\n</extraits>"


def generate(question: str, docs: list) -> str:
    response = get_claude().beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": CLAUDE_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        messages=[
            {
                "role": "user",
                "content": f"{format_context(docs)}\n\nQuestion : {question}",
            }
        ],
    )
    if response.stop_reason == "refusal":
        return REFUSAL_ANSWER
    text = "".join(b.text for b in response.content if b.type == "text")
    return text or REFUSAL_ANSWER


def answer_question(question: str) -> dict:
    in_scope, sensitivity = jev.triage(question)
    if in_scope < IN_SCOPE_THRESHOLD:
        return {
            "answer": "Je ne peux répondre qu'aux questions sur le Code pénal du Burundi.",
            "in_scope": in_scope,
            "grounded": None,
            "sensitivity": sensitivity,
            "sources": [],
        }

    docs = retriever.invoke(question)
    docs = jev.rerank(question, docs)
    if not docs:
        return {
            "answer": "Je ne trouve pas cette information dans le Code pénal.",
            "in_scope": in_scope,
            "grounded": None,
            "sensitivity": sensitivity,
            "sources": [],
        }
    answer = generate(question, docs)
    if answer == REFUSAL_ANSWER:
        return {
            "answer": answer,
            "in_scope": in_scope,
            "grounded": None,
            "sensitivity": sensitivity,
            "sources": [],
        }
    score = jev.grounded(question, docs, answer)

    if score < GROUNDED_THRESHOLD:
        answer += "\n\n⚠️ Réponse partiellement vérifiée : consultez le texte officiel."
    if sensitivity == "élevé":
        answer += "\n\nVotre situation semble sérieuse : consultez un avocat."

    return {
        "answer": answer,
        "in_scope": in_scope,
        "grounded": score,
        "sensitivity": sensitivity,
        "sources": [d.metadata.get("page") for d in docs],
    }
