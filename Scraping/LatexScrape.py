from sickle import Sickle
from sickle.models import Record
import time
import requests
import os

sickle = Sickle('http://export.arxiv.org/oai2')

params = {
    'metadataPrefix': 'arXiv',
    'set': 'cs',          
    'from': '2024-01-01',
    'until': '2024-12-31'
}

base_dir = '/Users/karthikdubba/Classes/Fall 2024/CSE 511/Project'

year = '2024'
year_dir = os.path.join(base_dir, year)
os.makedirs(year_dir, exist_ok=True)

paper_count = 0

records = sickle.ListRecords(**params)

for record in records:
    if isinstance(record, Record):
        metadata = record.metadata
        arxiv_id = metadata.get('id')[0]

        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        try:
            response = requests.get(source_url, timeout=10)
            response.raise_for_status()
            source_filename = f"{arxiv_id.replace('/', '_')}.tar.gz"
            source_path = os.path.join(year_dir, source_filename)
            with open(source_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded LaTeX source: {source_path}")
            paper_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Failed to download LaTeX source for {arxiv_id}: {e}")

        time.sleep(3)
    else:
        print("No more records.")
        break

print(f"Total LaTeX source files downloaded: {paper_count}")
