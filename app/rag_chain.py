import os

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
MAX_TOKENS = 16000

# Choix du modèle via .env : LLM_PROVIDER, LLM_MODEL et LLM_EFFORT (facultatifs).
# DeepSeek expose une API compatible Anthropic, d'où le même SDK ; Gemini a le sien.
DEFAULT_PROVIDER = "gemini"
PROVIDERS = {
    "gemini": {
        "base_url": None,
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.8-flash",
        "default_effort": "medium",  # thinking_level : low | medium | high
        "models": None,
    },
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5-5",
        "default_effort": "medium",
        "models": None,  # tout modèle Claude est accepté
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/anthropic",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-pro",
        "default_effort": "high",
        # DeepSeek remplace silencieusement un nom inconnu par deepseek-flash.
        "models": {"deepseek-v4-pro", "deepseek-flash"},
    },
}


def load_llm_config(env=os.environ) -> dict:
    provider = env.get("LLM_PROVIDER", "").strip().lower() or DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} inconnu. Valeurs possibles : {sorted(PROVIDERS)}"
        )
    spec = PROVIDERS[provider]
    model = env.get("LLM_MODEL", "").strip() or spec["default_model"]
    if spec["models"] is not None and model not in spec["models"]:
        raise ValueError(
            f"LLM_MODEL={model!r} invalide pour {provider}. "
            f"Valeurs possibles : {sorted(spec['models'])}"
        )
    effort = env.get("LLM_EFFORT", "").strip() or spec["default_effort"]
    return {"provider": provider, "model": model, "effort": effort, **spec}


LLM = load_llm_config()

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

_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get(LLM["api_key_env"])
        # Sans clé, le SDK se rabattrait sur ANTHROPIC_API_KEY : jamais vers un tiers.
        if LLM["provider"] != "anthropic" and not api_key:
            raise RuntimeError(f"{LLM['api_key_env']} manquant dans .env")
        if LLM["provider"] == "gemini":
            from google import genai  # import local : inutile pour les autres fournisseurs

            _llm_client = genai.Client(api_key=api_key)
        else:
            _llm_client = anthropic.Anthropic(
                api_key=api_key, base_url=LLM["base_url"]
            )
    return _llm_client


def format_context(docs) -> str:
    extraits = "\n\n".join(
        f"[page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )
    return f"<extraits>\n{extraits}\n</extraits>"


def generate(question: str, docs: list) -> str:
    user_content = f"{format_context(docs)}\n\nQuestion : {question}"
    client = get_llm_client()
    if LLM["provider"] == "gemini":
        interaction = client.interactions.create(
            model=LLM["model"],
            input=user_content,
            system_instruction=SYSTEM_PROMPT,
            generation_config={
                "thinking_level": LLM["effort"],
                "max_output_tokens": MAX_TOKENS,
            },
            store=False,  # Google conserve les échanges par défaut
        )
        return interaction.output_text or REFUSAL_ANSWER

    request = {
        "model": LLM["model"],
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "output_config": {"effort": LLM["effort"]},
        "messages": [{"role": "user", "content": user_content}],
    }
    if LLM["provider"] == "anthropic":
        # Repli serveur si un classifieur refuse ; propre à l'API Anthropic.
        response = client.beta.messages.create(
            **request,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
        )
    else:
        response = client.messages.create(**request)
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
