"""Optional source retrieval; normal training uses the included attributed CSV."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, SOURCE_URL

def main():
    metadata = json.loads((ROOT / 'data/source.json').read_text())
    with urlopen(SOURCE_URL, timeout=90) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != metadata['source_zip_sha256']:
        raise ValueError('Source archive changed; review it before replacing the committed dataset.')
    with ZipFile(BytesIO(payload)) as archive:
        with archive.open('DryBeanDataset/Dry_Bean_Dataset.xlsx') as sheet:
            frame = pd.read_excel(sheet)
    destination = ROOT / 'data/raw/dry_bean.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f'Wrote {len(frame)} rows to {destination}')

if __name__ == '__main__':
    main()
