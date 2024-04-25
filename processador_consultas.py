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


log.info('Executando processador de consultas...\n')

pc_config_file = configs_dir/'PC.CFG'
pc_config = config_parser(pc_config_file)

queries_source_file = data_dir/pc_config['LEIA'][0]
queries_dest_file = results_dir/pc_config['CONSULTAS'][0]
expected_dest_file = results_dir/pc_config['ESPERADOS'][0]


log.info(f'Lendo arquivo de consultas em {queries_source_file}')
queries = ET.parse(queries_source_file)

queries_dict = {'QueryNumber':[], 'QueryText':[]}

start = time.time()
for query in queries.findall('QUERY'):
    query_number = query.find('QueryNumber').text
    query_text = query.find('QueryText').text
    queries_dict['QueryNumber'].append(query_number)
    queries_dict['QueryText'].append(query_text)

log.info(f'Número de consultas encontradas: {len(queries_dict['QueryNumber'])}\n')

log.info('Processando texto das consultas...')
queries_text = TextPreprocessor(queries_dict['QueryText'])
queries_dict['QueryText'] = queries_text \
                            .remove_escape_sequences() \
                            .remove_accents() \
                            .remove_punctuation() \
                            .remove_stopwords() \
                            .to_upper() \
                            .data
end = time.time()
log.info('Processamento de texto concluído.\n')
log.info(f'Consultas processadas em {(end-start) * 1000:.2f}ms.\n')

to_csv(queries_dest_file, queries_dict)

# ------------------

def score_to_votes(score: str) -> int:
    num_votes = 0
    for digit in score:
        num_votes += int(digit)
    return num_votes

log.info(f'Criando base de dados dos resultados esperados...')

expected_dict = {'QueryNumber':[], 'DocNumber':[], 'DocVotes':[]}

for query in queries.findall('QUERY'):
    query_number = query.find('QueryNumber').text
    for item in query.find('Records'):
        expected_dict['QueryNumber'].append(query_number)
        expected_dict['DocNumber'].append(item.text)
        expected_dict['DocVotes'].append(score_to_votes(item.get('score')))

to_csv(expected_dest_file, expected_dict)