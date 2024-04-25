import xml.etree.ElementTree as ET
from pathlib import Path
import logging as log
import time

from utils import config_parser, to_csv, TextPreprocessor

logger = log.getLogger()
logger.setLevel(log.INFO)

data_dir = Path('data')
results_dir = Path('RESULT')
configs_dir = Path('configs')

log.info('Executando gerador de lista invertida...\n')

gli_config_file = configs_dir/'GLI.CFG'
gli_config = config_parser(gli_config_file)
gli_source_files = gli_config['LEIA']
gli_dest_file = results_dir/gli_config['ESCREVA'][0]


log.info(f'Lendo diretório de documentos em /{data_dir}')

docs_dict = {'DocNumber':[], 'Abstract': []}

for file in gli_source_files:

    log.info(f'Lendo arquivo {file}...')
    docs = ET.parse(data_dir/file)

    doc_count = 0
    for doc in docs.findall('RECORD'):
        doc_number = doc.find('RECORDNUM').text
        abstract = doc.find('ABSTRACT')
        if abstract is None:
            abstract = doc.find('EXTRACT')
        if abstract is not None:
            doc_count += 1
            docs_dict['DocNumber'].append(int(doc_number.strip()))
            docs_dict['Abstract'].append(abstract.text)
    log.info(f'Documentos encontrados com "Abstract" ou "Extract": {doc_count}')

log.info(f'Total de documentos encontrados com "Abstract" ou "Extract": {len(docs_dict['DocNumber'])}\n')

log.info('Processando texto dos documentos...')
abstracts_text = TextPreprocessor(docs_dict['Abstract'])
docs_dict['Abstract'] = abstracts_text \
                        .remove_escape_sequences() \
                        .remove_accents() \
                        .remove_punctuation() \
                        .remove_stopwords() \
                        .to_upper() \
                        .data
log.info('Processamento de texto concluído.\n')


log.info(f'Iniciando construção da lista invertida...')

gli_dict = {}
start = time.time()
for doc_number, abstract in zip(*docs_dict.values()):
    words = abstract.split()
    for word in words:
        if word in gli_dict:
            gli_dict[word].append(doc_number)
        else:
            gli_dict[word] = [doc_number]
end = time.time()
log.info(f'Número de palavras únicas: {len(gli_dict)}')
log.info(f'Lista invertida construída em {(end-start) * 1000:.2f}ms.\n')

to_csv(gli_dest_file, {'Word':list(gli_dict.keys()), 'WordDocs':list(gli_dict.values())}, headers=False)