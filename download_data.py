# Run once at the command line
# pip install --upgrade numerapi

# python3
import os
import pandas as pd

import numerapi

DATA_DIRECTORY = os.path.expanduser("~/numerai_data/")

napi = numerapi.NumerAPI(verbosity="info")
current_round = napi.get_current_round()
current_data_path = f'{DATA_DIRECTORY}numerai_dataset_{current_round}'

current_data_needs_downloaded = not os.path.isdir(current_data_path)

if current_data_needs_downloaded:
    napi.download_current_dataset(dest_path=DATA_DIRECTORY)

# training_data = pd.read_csv(os.path.join(current_data_path, 'numerai_training_data.csv'))

# training_data.apply(lambda x: x.autocorr())
