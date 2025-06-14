from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv() 


loader = PyPDFLoader("Burundi_Code_2017_penal.pdf")
data = loader.load()  # entire PDF is loaded as a single Document

# split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600)
docs = text_splitter.split_documents(data)
# print(docs[200])
# print("================================================")
print("Total number of documents: ",len(docs))


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("hello, world!")
# vector[:5]


vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# retrieved_docs = retriever.invoke("Selon l'article 1, qu'est-ce qu'une infraction ?")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",temperature=1, max_tokens=500)


system_prompt = (
    "Vous êtes un assistant juridique spécialisé dans le Code pénal du Burundi. "
    "Votre rôle est de fournir des informations précises et contextuelles sur le droit pénal burundais. "
    
    "RÈGLES DE RÉPONSE : "
    "• Utilisez UNIQUEMENT les extraits de contexte fournis pour formuler votre réponse "
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
    
    "CONTEXTE DU CODE PÉNAL : "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def answer_question(question: str) -> str:
    response = rag_chain.invoke({"input": question})
    return response["answer"]

# response = rag_chain.invoke({"input": "Est-ce que la loi fait une différence entre voler un portefeuille dans un sac et cambrioler une maison en cassant une fenêtre ? Comment les punitions changent-elles ?"})
# print(response["answer"])