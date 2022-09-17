import functions.documents_process as dp
import numpy as np


def calc_dcg_k(relevance):
    result = 0
    for i in range(len(relevance)):
        log = np.log2(1 + (i + 1))
        result += (relevance[i] / log)
    return result


def calc_ndcg_k(relevance, true_relevance):
    return calc_dcg_k(relevance) / calc_dcg_k(true_relevance)


relevance = [0, 3, 3]
true_relevance = [4, 3, 3]

calc_ndcg_k(relevance, true_relevance)
# ndcg_score(np.asarray([true_relevance]), np.asarray([relevance]))


def metrics_dcg_k(dataset, documents):
    return ""


def generate_metrics_dcg_k(dataset, documents, engine):
    query_id = []
    true_dcg_k = []
    nir_dcg_k = []
    nir_ndcg_k = []

    process = dp.DocumentProcess(dataset, documents, engine)

    for id in range(1, 256):

        true_relevance = process.get_true_relevance(query_id=id)
        relevance = process.get_query_relevance(query_id=id)

        if dp.sum_relevance(relevance) > 0:
            query_id.append(id)
            true_dcg_k.append(calc_dcg_k(true_relevance))
            nir_dcg_k.append(calc_dcg_k(relevance))
            nir_ndcg_k.append(calc_ndcg_k(relevance, true_relevance))
    if engine == "ELS":
        return {
            'query_id': query_id,
            'true_dcg_k': true_dcg_k,
            'els_dcg_k': nir_dcg_k,
            'els_ndcg_k': nir_ndcg_k,
        }
    else:
        return {
            'query_id': query_id,
            'true_dcg_k': true_dcg_k,
            'nir_dcg_k': nir_dcg_k,
            'nir_ndcg_k': nir_ndcg_k,
        }


def merge(nir_ndcg, els_ndcg):
    i = 0

    vquery_id = []
    vtrue_dcg_k = []
    nir_dcg_k = []
    nir_ndcg_k = []
    els_dcg_k = []
    els_ndcg_k = []

    i = 0
    for nir_id in nir_ndcg['query_id']:

        query_id = nir_ndcg['query_id'][nir_id]
        for els_id in els_ndcg['query_id']:
            if i >= len(els_ndcg['query_id']):
                continue
            if els_ndcg['query_id'][els_id] == query_id:
                vquery_id.append(query_id)
                vtrue_dcg_k.append(nir_ndcg['true_dcg_k'][str(i)])
                nir_dcg_k.append(nir_ndcg['nir_dcg_k'][str(i)])
                nir_ndcg_k.append(nir_ndcg['nir_ndcg_k'][str(i)])
                els_dcg_k.append(els_ndcg['els_dcg_k'][str(i)])
                els_ndcg_k.append(els_ndcg['els_ndcg_k'][str(i)])
        i += 1

    return {
        'query_id': vquery_id,
        'true_dcg_k': vtrue_dcg_k,
        'nir_dcg_k': nir_dcg_k,
        'els_dcg_k': els_dcg_k,
        'nir_ndcg_k': nir_ndcg_k,
        'els_ndcg_k': els_ndcg_k,
    }


def get_query_documents(query_id, documents, engine):
    temp_documents = []
    for doc in documents:
        if str(query_id) != doc['query_id']:
            continue
        query_results = {}
        if engine == "NIR":
            query_results = doc['result']['queryResults']
        if engine == "ELS":
            query_results = documents[0]['result']['hits']['hits']

        for result in query_results:
            temp_documents.append(result)
    return temp_documents


def is_contain_document(doc_id, qrels):
    for qrel in qrels:
        if (qrel.doc_id == str(doc_id)):
            return True
    return False


def get_qrels(query_id, dataset):
    temp_qrels = []
    for qrel in dataset.qrels_iter():
        if qrel.query_id == str(query_id):
            temp_qrels.append(qrel)
    return temp_qrels


def get_total_retrieved_relevant(query_id, documents, dataset, engine):
    count = 0
    qrels = get_qrels(query_id, dataset)
    query_documents = get_query_documents(query_id, documents, engine)

    for doc in query_documents:
        document_id = {}
        if engine == "NIR":
            document_id = doc['document']['id']
        if engine == "ELS":
            document_id = doc['_id']

        if is_contain_document(document_id, qrels):
            count += 1

    return count


def get_total_relevant(query_id, dataset):
    count = 0
    for qrel in dataset.qrels_iter():
        if qrel.query_id == str(query_id):
            count += 1
    return count


# precisão = 15 relevantes recuperados / 25 total recuperado
def calc_precision(retrieved_relevant, total_retrieved):
    if retrieved_relevant == 0:
        return 0
    return retrieved_relevant / total_retrieved


# recall = 15 recuperados relevantes / 20 total relevante
def calc_recall(retrieved_relevant, total_relevant):
    if retrieved_relevant == 0:
        return 0
    return retrieved_relevant / total_relevant


def f_score(recall, precision):
    n = 2 * (precision * recall)
    d = precision + recall
    return n / d


def dict_to_list(dict):
    temp = []
    for key in dict:
        temp.append(dict[key])
    return temp


def recall_precision(dataset, documents, engine):
    query_ids = []
    total_relevants = []
    total_retrieveds = []
    total_retrieved_relevants = []
    recall = []
    precision = []
    f_scores = []
    for id in range(1, 256):

        total_retrieved_relevant = get_total_retrieved_relevant(
            id, documents, dataset, engine)

        if total_retrieved_relevant == 0:
            continue

        query_ids.append(id)
        total_relevant = get_total_relevant(id, dataset)
        total_retrieved = len(get_query_documents(id, documents, engine))

        total_relevants.append(total_relevant)
        total_retrieveds.append(total_retrieved)
        total_retrieved_relevants.append(total_retrieved_relevant)

        value_recall = calc_recall(total_retrieved_relevant, total_relevant)
        value_precision = calc_precision(
            total_retrieved_relevant, total_retrieved)
        recall.append(value_recall)
        precision.append(value_precision)
        f_scores.append(f_score(value_recall, value_precision))

    return {
        "query_ids": query_ids,
        "total_relevants": total_relevants,
        "total_retrieveds": total_retrieveds,
        "total_retrieved_relevants": total_retrieved_relevants,
        "recall": recall,
        "precision": precision,
        "f_scores": f_scores
    }
