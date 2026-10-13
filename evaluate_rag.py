import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv(dotenv_path="/Users/revathy_santhosh/rag-project/.env")

Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("LLM and embedding model configured.")

import glob
import fitz  # this is PyMuPDF's import name
from llama_index.core import Document

documents = []
for pdf_path in glob.glob("/Users/revathy_santhosh/rag-project/data/*.pdf"):
    pdf = fitz.open(pdf_path)
    file_name = pdf_path.split("/")[-1]
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()
        if text.strip():  # skip blank pages
            documents.append(Document(text=text, metadata={"file_name": file_name, "page": page_num}))

print(f"Loaded {len(documents)} page-documents from {len(set(d.metadata['file_name'] for d in documents))} files")
for fname in sorted(set(d.metadata["file_name"] for d in documents)):
    total_chars = sum(len(d.text) for d in documents if d.metadata["file_name"] == fname)
    print(f"- {fname}: {total_chars} chars")

    import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

chroma_client = chromadb.PersistentClient(path="/Users/revathy_santhosh/rag-project/chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_project")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
print("Index built and stored in Chroma.")

query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("What does the HBR article say about how Apple organizes for innovation?")
print(response)

print("\n--- Sources used ---")
for node in response.source_nodes:
    print(f"- {node.metadata.get('file_name')} (page {node.metadata.get('page')}) — score: {node.score:.3f}")

eval_questions = [
    {"question": "According to the HBR article, how is Apple organized to support innovation?",
     "ground_truth": "Apple is organized around functional expertise rather than by product lines or business units, with senior leaders who are deep experts in their discipline, enabling fast, high-quality decisions at scale."},
    {"question": "What is the last date for submitting the B.Des application form for NID's 2026-27 admissions?",
     "ground_truth": "11:59 pm (midnight) on Monday, 01 December 2025."},
    {"question": "When is the Design Aptitude Test (DAT) Prelims scheduled for NID's B.Des 2026-27 admissions?",
     "ground_truth": "Sunday, 21 December 2025."},
    {"question": "What NYIT office publishes the 'Coming to New York' guide for international students?",
     "ground_truth": "NYIT's Office of International Education."},
    {"question": "What visa statuses does NYIT's international student guide cover for obtaining student status?",
     "ground_truth": "F-1 or J-1 student status."},
    {"question": "Who prepared the Edward Guiliano Global Fellowship 2023/2024 Annual Report?",
     "ground_truth": "Askia VanOmmeren, Assistant Director of Experiential Education."},
    {"question": "According to the AI Engineering book's RAG chapter, what are the main steps in a RAG system's workflow?",
     "ground_truth": "A RAG system loads documents, chunks them, embeds the chunks, retrieves the most relevant chunks for a given query, and generates an answer using that retrieved context."},
]
eval_data = []

for item in eval_questions:
    response = query_engine.query(item["question"])
    eval_data.append({
        "user_input": item["question"],
        "response": str(response),
        "retrieved_contexts": [node.text for node in response.source_nodes],
        "reference": item["ground_truth"],
    })
    print(f"done: {item['question'][:60]}...")

print(f"\nCollected {len(eval_data)} Q&A pairs.")

#eval_data = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

#for item in eval_questions:
    #response = query_engine.query(item["question"])
    #eval_data["user_input"].append(item["question"])
    #eval_data["response"].append(str(response))
    #eval_data["retrieved_contexts"].append([node.text for node in response.source_nodes])
    #eval_data["reference"].append(item["ground_truth"])
    #print(f"done: {item['question'][:60]}...")

#print(f"\nCollected {len(eval_data['user_input'])} Q&A pairs.")

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.run_config import RunConfig

ragas_llm = LlamaIndexLLMWrapper(Settings.llm)
ragas_embeddings = LlamaIndexEmbeddingsWrapper(Settings.embed_model)

dataset = EvaluationDataset.from_dict(eval_data)

my_run_config = RunConfig(timeout=120, max_retries=3, max_wait=20, max_workers=2)

result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ],
    run_config=my_run_config,
    raise_exceptions=True,
    show_progress=True,
)

print(result)
df = result.to_pandas()
df.to_csv("ragas_eval_results.csv", index=False)
print("\nSaved to ragas_eval_results.csv")