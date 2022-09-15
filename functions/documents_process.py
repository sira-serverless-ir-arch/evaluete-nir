class DocResult:
    def __init__(self, query_id, document_id, relevance, seq):
        self.query_id = query_id
        self.document_id = document_id
        self.relevance = relevance
        self.seq = seq


class DocumentProcess:

    def __init__(self, dataset, documents, engine):
        self.dataset = dataset
        self.documents = documents
        self.engine = engine

    def get_documents(self, query_id):
        temp_documents = []
        for doc in self.documents:
            if str(query_id) != doc['query_id']:
                continue
            query_results = {}
            if self.engine == "NIR":
                query_results = doc['result']['queryResults']
            if self.engine == "ELS":
                query_results = self.documents[0]['result']['hits']['hits']

            for result in query_results:
                temp_documents.append(result)
        return temp_documents

    def get_qrels(self, query_id):
        temp_qrels = []
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id == str(query_id):
                temp_qrels.append(qrel)

        return temp_qrels

    def get_doc_relevance(self, doc_id, qrels):
        for qrel in qrels:
            if (qrel.doc_id == str(doc_id)):
                return qrel.relevance
        return 0

    def get_query_result_docs(self, query_id, query_documents):

        temp_docs = []
        qrels = self.get_qrels(query_id)

        seq = 0
        for doc in query_documents:
            document_id = {}
            if self.engine == "NIR":
                document_id = doc['document']['id']
            if self.engine == "ELS":
                document_id = doc['_id']

            relevance = self.get_doc_relevance(document_id, qrels)
            seq += 1
            temp_docs.append(DocResult(query_id, document_id, relevance, seq))

        return temp_docs

    def get_query_relevance(self, query_id):
        result_docs = self.get_query_result_docs(
            query_id=query_id,
            query_documents=self.get_documents(query_id=query_id))

        relevance = []
        for x in result_docs:
            relevance.append(x.relevance)
            #print("K:", i, " ",x.document_id, " ", x.relevance)
        return relevance

    def get_true_relevance(self, query_id):
        true_relevance = []
        for qrel in self.get_qrels(query_id=query_id):
            true_relevance.append(qrel.relevance)
        return true_relevance


def sum_relevance(relevances):
    sm = 0
    for x in relevances:
        sm += x
    return sm
