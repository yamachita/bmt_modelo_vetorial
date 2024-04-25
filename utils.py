import xml.etree.ElementTree as ET
from pathlib import Path
import logging as log
import unicodedata
import string
import csv
import re

import nltk
nltk.download('stopwords', quiet=True)

def config_parser(config_file_path: str|Path) -> dict:
    '''Retorna dict no formato {instrução:[arquivos]}'''
    log.info(f'Lendo arquivo de configuração {config_file_path}')

    with open(config_file_path, 'r') as config_file:
        lines = config_file.read().split('\n')

    instructions = {}
    for k,v in [i.split('=') for i in lines]:
        k = k.strip()
        v = v.strip()
        if k in instructions:
            instructions[k].append(v)
        else:
            instructions[k] = [v]

    log.info(f'Configurações encontradas em {config_file_path.name}:\n{instructions}\n')
    
    return instructions


def to_csv(csv_file_path: str|Path, data: dict, headers: bool = True) -> None:
    log.info(f'Salvando dados no arquivo {csv_file_path}')

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        if headers:
            writer.writerow(list(data.keys()))

        lines = list(zip(*data.values()))
        writer.writerows(lines)

        if headers:
            log.info(f'Cabeçalho do arquivo salvo:\n{list(data.keys())}')


class TextPreprocessor:
    def __init__(self, data: list[str]):
        self.data = data

    def to_upper(self):
        log.info(f'Convertendo letras para maiúsculas...')
        self.data = list(map(lambda x: x.upper(), self.data))
        return self
    
    def remove_accents(self):
        log.info(f'Removendo acentos...')
        self.data = list(map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'), self.data))
        return self
    
    def remove_stopwords(self, stopwords: list[str] = nltk.corpus.stopwords.words('english')):
        log.info(f'Removendo stopwords...')
        self.data = list(map(
            lambda x: ' '.join([ word for word in x.split() if word.lower() not in stopwords]),
            self.data))
        return self
    
    def remove_punctuation(self):
        log.info(f'Removendo pontuação...')
        self.data = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), self.data))
        return self
    
    def remove_escape_sequences(self):
        log.info(f'Removendo sequências de escapes...')
        self.data = list(map(lambda x: re.sub(r'\s+',' ',x).strip(), self.data))
        return self
    
    def stemming(self, stemmer: nltk.stem.api.StemmerI = nltk.stem.porter.PorterStemmer()):
        log.info(f'Aplicando stemming...')
        self.data = list(map(lambda x: ' '.join([stemmer.stem(s) for s in x.split()]), self.data))
        return self