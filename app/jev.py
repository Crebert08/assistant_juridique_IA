from typesafe_sdk import Noul, Score, TypeSafeClient

RELEVANCE_ORDER = {"non": 0, "partiellement": 1, "oui": 2}

_client = None


def get_client():
    global _client
    if _client is None:
        _client = TypeSafeClient()
    return _client


def triage(question: str) -> tuple[float, str]:
    result = get_client().system_one(
        question,
        {
            "in_scope": Noul(
                instructions="La question porte-t-elle sur le droit pénal burundais ?"
            ),
            "sensitivity": Score(
                instructions="Gravité de la situation décrite (détention, violence, mineurs) ?",
                criteria=["faible", "moyen", "élevé"],
            ),
        },
    )
    return result.nouls["in_scope"].noul, result.scores["sensitivity"].score


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
                    criteria=["non", "partiellement", "oui"],
                )
            },
        )
        relevance = result.scores["relevance"]
        ranked.append(
            (RELEVANCE_ORDER.get(relevance.score, 0), relevance.confidence, doc)
        )
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
