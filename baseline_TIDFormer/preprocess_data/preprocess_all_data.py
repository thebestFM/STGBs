import os

for name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci']:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
