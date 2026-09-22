from typesafe_sdk import Noul, Score, TypeSafeClient

# Score renvoie une moyenne pondérée des indices de niveaux (0 .. len-1), pas un libellé.
SENSITIVITY_LEVELS = ["faible", "moyen", "élevé"]
RELEVANCE_LEVELS = ["non", "partiellement", "oui"]

# Sans ce contexte, Jev rejette les questions qui ne mentionnent pas le Burundi.
APP_CONTEXT = (
    "Assistant juridique sur le Code pénal du Burundi. Les utilisateurs sont au "
    "Burundi : toute question de droit pénal concerne le droit burundais, même "
    "si le pays n'est pas mentionné."
)

_client = None


def get_client():
    global _client
    if _client is None:
        _client = TypeSafeClient()
    return _client


def triage(question: str) -> tuple[float, str]:
    result = get_client().system_one(
        {"contexte": APP_CONTEXT, "question": question},
        {
            "in_scope": Noul(
                instructions=(
                    "La question porte-t-elle sur le droit pénal : infractions, "
                    "crimes, délits, peines, responsabilité pénale ou procédure "
                    "pénale ?"
                )
            ),
            "sensitivity": Score(
                instructions="Gravité de la situation décrite (détention, violence, mineurs) ?",
                criteria=SENSITIVITY_LEVELS,
            ),
        },
    )
    level = round(result.scores["sensitivity"].score)
    level = min(max(level, 0), len(SENSITIVITY_LEVELS) - 1)
    return result.nouls["in_scope"].noul, SENSITIVITY_LEVELS[level]


def rerank(question: str, docs: list, keep: int = 4) -> list:
    if not docs:
        return []
    ranked = []
    for doc in docs:
        result = get_client().system_one(
            {"question": question, "extrait": doc.page_content},
            {
                "relevance": Score(
                    instructions="Cet extrait aide-t-il à répondre à la question ?",
                    criteria=RELEVANCE_LEVELS,
                )
            },
        )
        relevance = result.scores["relevance"]
        ranked.append((relevance.score, relevance.confidence, doc))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [doc for _, _, doc in ranked[:keep]]


def grounded(question: str, docs: list, answer: str) -> float:
    result = get_client().system_one(
        {
            "question": question,
            "articles": "\n\n".join(d.page_content for d in docs),
            "reponse": answer,
        },
        {
            "grounded": Noul(
                instructions="La réponse est-elle entièrement justifiée par les articles fournis ?"
            )
        },
    )
    return result.nouls["grounded"].noul
