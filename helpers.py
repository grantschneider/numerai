def numeric_era_converter(era_string):
  era_string_formatted = era_string.replace('era', '')
  era_string_formatted = era_string_formatted.replace('X', '999')
  return int(era_string_formatted)
