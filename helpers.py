import os

DATA_DIRECTORY = os.path.expanduser("~/numerai_data/")

POSSIBLE_MARKET_NAMES = [
  'target_bernie', 'target_charles', 
  'target_elizabeth', 'target_jordan', 
  'target_ken'
]

def numeric_era_converter(era_string):
  era_string = era_string.replace('era', '')
  era_string = era_string.replace('X', '999')
  return int(era_string)
