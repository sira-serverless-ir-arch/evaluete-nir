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
