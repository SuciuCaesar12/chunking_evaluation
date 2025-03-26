from pathlib import Path
import pandas as pd
import json
import platform

from utils.chunking import *


def load_questions_df(questions_csv_path: Path, corpora_id: str):
    assert questions_csv_path.exists(), f'File {questions_csv_path} does not exist'

    questions_df = pd.read_csv(questions_csv_path)
    # for our use case, keep only the questions from the specified corpus
    questions_df = questions_df[questions_df['corpus_id'] == corpora_id]
    questions_df['references'] = questions_df['references'].apply(json.loads)
    
    return questions_df


def load_corpora(path: Path):
    assert path.exists(), f'File {path} does not exist'

    if platform.system() == 'Windows':
        with open(path, 'r', encoding='utf-8') as file:
            corpus = file.read()
    else:
        # Use default encoding on other systems
        with open(path, 'r') as file:
            corpus = file.read()
    
    return corpus
