import sys
sys.dont_write_bytecode = True
from langchain_community.document_loaders import UnstructuredPDFLoader

class RAGPipeline():
  def __init__(self, vectorstore_db):
    self.vectorstore = vectorstore_db

  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results


  def __retrieve_docs_id__(self, question: str, k=10):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["source"]): score for doc, score in docs_score}
    return docs_score


  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents


  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
    retrieved_documents = []
    for doc_path in retrieved_ids:
        loader = UnstructuredPDFLoader(doc_path, mode="single", strategy="fast") 
        retrieved_documents.extend(loader.load())
    for i in range(len(retrieved_documents)):
      retrieved_documents[i] = "Applicant File " + retrieved_ids[i] + "\n" + retrieved_documents[i].page_content
    return retrieved_documents