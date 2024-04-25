from typing import Callable
from pathlib import Path
import logging as log
import json
import time
import math
import csv

from utils import config_parser

logger = log.getLogger()
logger.setLevel(log.INFO)

results_dir = Path('RESULT')
configs_dir = Path('configs')


class Indexador:

    def __init__(self, inverted_index_path: Path|str):

        self.tf_normalization_factor_func = lambda x: max(x.values())

        log.info('Criando modelo vetorial...')
        start = time.time()
        self.vector_model_tfidfs = self._create_vector_model_tfidf(inverted_index_path)
        end = time.time()
        log.info(f'Modelo vetorial criado em {(end-start) * 1000:.2f}ms.\n')
            

    def save_vector_model_to_json(self, json_file_path: Path|str) -> None:
        log.info(f'Salvando modelo em {json_file_path}\n')
        json_model = {'vector_model_tfidf': self.vector_model_tfidfs,
                      'word_index': self.word_index,
                      'word_idfs': self.word_idfs}
        
        with open(json_file_path, 'w') as json_file:
            json.dump(json_model, json_file)

    def _create_vector_model_tfidf(self, inverted_index_path: Path|str) -> dict:

        inverted_index = self._load_inverted_index_from_csv(inverted_index_path)
        inverted_index = self._preprocess_inverted_index(inverted_index)
        self.word_index = self._create_word_index(inverted_index)
        return self._create_tfidfs(inverted_index)


    def _load_inverted_index_from_csv(self, inverted_index_path: Path|str) -> dict:
        '''Lê o arquivo da lista invertida e armazena em um dict'''
        log.info(f'Carregando lista invertida de /{inverted_index_path}')
        inverted_index_dict = {}

        with open(inverted_index_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for word, word_docs in csv_reader:
                inverted_index_dict[word] = list(map(lambda x: int(x), word_docs[1:-1].split(',')))
        return inverted_index_dict
    
    def _preprocess_inverted_index(self, inverted_index: dict) -> dict:
        '''Somente palavras com tamanho >= 2 e sem números'''

        log.info('Normalizando lista (somente palavras com tamanho >=2 e sem dígitos)...')
        words = list(inverted_index.keys())
        for word in words:
            if len(word) < 2 or any(ch.isdigit() for ch in word):
                del inverted_index[word]

        log.info(f'Número de palavras únicas: {len(inverted_index)}')
        return inverted_index
    
    def _create_word_index(self, inverted_index: dict) -> dict:
        '''Dicionário para mapear cada palavra a um índice para o vetor de representação dos documentos'''
        return dict(zip(list(inverted_index.keys()), list(range(len(inverted_index)))))
    
    
    def _create_doc_vectors_freq(self, inverted_index: dict) -> dict:
        doc_index_dict = {}
        for word, word_docs in inverted_index.items():
            for doc in word_docs:
                word_index = self.word_index[word]
                if doc not in doc_index_dict:
                    doc_index_dict[doc] = {}
                if word_index not in doc_index_dict[doc]:
                    doc_index_dict[doc][word_index] = 1
                else:
                    doc_index_dict[doc][word_index] += 1
        return doc_index_dict

    def _create_idfs(self, inverted_index: dict) -> dict:
        '''idf de cada palavra do vocabulário'''

        flat_docs = []
        for row in list(inverted_index.values()):
            flat_docs += row
        number_of_docs = len(set(flat_docs))

        log.info(f'Número de documentos: {number_of_docs}')

        idfs = {}
        for word, word_docs in inverted_index.items():
            df = len(set(word_docs))
            idf = math.log10(number_of_docs/df)
            idfs[self.word_index[word]] = idf
        return idfs
    
    def set_tf_normalization_factor_func(self, tf_normalization_factor_func: Callable[[dict], int]):
        self.tf_normaliztion_factor_func = tf_normalization_factor_func
    
    def _create_tfidfs(self, inverted_index: dict):
        log.info('Criando modelo de representação tf-idf com o seguinte formato para cada documento:\n'\
                 '{índice_palavra1: tfidf_palavra1, índice_palavra2: tfidf_palavra2, ...}')
        self.word_idfs = self._create_idfs(inverted_index)
        doc_vectors_freq = self._create_doc_vectors_freq(inverted_index)

        doc_index_tfidf = {}

        for doc, word_freqs in doc_vectors_freq.items():
            doc_index_tfidf[str(doc)] = {}
            tf_normalization_factor = self.tf_normalization_factor_func(word_freqs)
            for word, tf in word_freqs.items():
                ntf = tf/tf_normalization_factor
                doc_index_tfidf[str(doc)][str(word)] = ntf * self.word_idfs[word]
        return doc_index_tfidf
    

index_config_file = configs_dir/'INDEX.CFG'
index_config = config_parser(index_config_file)
inverted_index_path = results_dir/index_config['LEIA'][0]
vector_model_json_path = results_dir/index_config['ESCREVA'][0]

log.info('Executando indexador...\n')
idx = Indexador(inverted_index_path)
idx.save_vector_model_to_json(vector_model_json_path)