import sys
sys.dont_write_bytecode = True
from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain.retrievers import EnsembleRetriever
# from llama_index.retrievers.bm25 import BM25Retriever
class RAGPipeline():
    def __init__(self, vectorstore_db):
        self.vectorstore = vectorstore_db

    def __retrieve_docs_id__(self, question: str, k=5):
        # vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k":5, "score_threshold": 0.75}, search_type="mmr")
        docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
        # print("\n\n\ndocs_score: ", docs_score)
        docs_score = {str(doc.metadata["source"]): 1-score for doc, score in docs_score}
        # # print("docs score:", docs_score)
        # bm25_retriever = BM25Retriever.from_defaults(
        #     docstore=self.vectorstore, similarity_top_k=5
        # )
        # ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5,0.5])
        # retrieved_docs = ensemble_retriever.invoke(question)
        # print("ENSEMBLED RETRIEVED DOCS", retrieved_docs)
        return docs_score

    def retrieve_id_and_rerank(self, question: str, similarity_threshold: float = 0.5):
        document_scores = self.__retrieve_docs_id__(question)
        
        # Enhanced Filtering: Ensure there's a minimum score difference for valid matches
        filtered_documents = {}
        for doc_id, score in document_scores.items():
            if score >= similarity_threshold:
                filtered_documents[doc_id] = score

        return filtered_documents

    def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
        retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
        retrieved_documents = []
        for doc_path in retrieved_ids:
            loader = UnstructuredPDFLoader(doc_path, mode="single", strategy="fast")
            retrieved_documents.extend(loader.load())
        for i in range(len(retrieved_documents)):
            retrieved_documents[i] = "Applicant File " + retrieved_ids[i] + "\n" + retrieved_documents[i].page_content
        return retrieved_documents
