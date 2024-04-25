from pathlib import Path
import logging as log
import json
import time
import math
import csv

from utils import config_parser, to_csv

logger = log.getLogger()
logger.setLevel(log.INFO)

results_dir = Path('RESULT')
configs_dir = Path('configs')


class BuscadorModeloVetorial:

    def __init__(self, vector_model_json_path: Path|str|None = None):
        
        with open(vector_model_json_path) as json_file:
            model_dict = json.load(json_file)

        self.vector_model_tfidfs = model_dict['vector_model_tfidf']
        self.word_index = model_dict['word_index']
        self.word_idfs = model_dict['word_idfs']
        log.info(f'Modelo carregado de {vector_model_json_path}\n')


    def _norm2(self, vec: dict) -> float:
        return math.sqrt(sum([x**2 for x in vec.values()]))

    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        vec1_norm = self._norm2(vec1)
        vec2_norm = self._norm2(vec2)

        dot_product = 0
        for w_index, w_tfidf in vec1.items():
            dot_product += w_tfidf * vec2.get(str(w_index), 0)

        return dot_product/(vec1_norm * vec2_norm)

    def find_docs(self, query: dict) -> dict:
        distances = {}
        for doc, tfidf_vector in self.vector_model_tfidfs.items():
            distances.update({doc: 1-self._cosine_similarity(query, tfidf_vector)})
        return dict(sorted(distances.items(), key=lambda x: x[1]))   
    
    def query_setup(self, query: list) -> dict:
        query_dict = {}
        for word in query:
            w_index = self.word_index.get(word)
            if w_index is not None:
                query_dict[w_index] = 1
        return query_dict
    

busca_config_file = configs_dir/'BUSCA.CFG'
busca_config = config_parser(busca_config_file)
vector_model_json_path = results_dir/busca_config['MODELO'][0]
queries_file = results_dir/busca_config['CONSULTAS'][0]
results_file = results_dir/busca_config['RESULTADOS'][0]

log.info('Executando buscador...\n')
vm = BuscadorModeloVetorial(vector_model_json_path)

log.info(f'Carregando consultas de {queries_file}...')
queries_dict = {}
header = True
with open(queries_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for query_number, query_text in csv_reader:
        if header:
            header = False
            continue
        queries_dict[query_number] = vm.query_setup(query_text.split())

log.info(f'Consultas encontradas: {len(queries_dict)}\n')

results_dict = {'Consulta': [], 'Ranking': []}

start = time.time()
for query_number, query in queries_dict.items():
    docs_ranking = vm.find_docs(query)
    result_rank = []
    for i, (doc_number, distance ) in enumerate(docs_ranking.items(), 1):
        result_rank.append((i, doc_number, distance))
    results_dict['Consulta'].append(query_number)
    results_dict['Ranking'].append(result_rank)
end = time.time()
log.info(f'Buscas realizadas em {(end-start) * 1000:.2f}ms.')
log.info(f'Tempo médio por busca {((end-start) * 1000)/len(queries_dict):.2f}ms.\n')

to_csv(results_file, results_dict, headers=False)