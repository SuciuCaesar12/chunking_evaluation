from itertools import product
from pathlib import Path
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import logging
from tqdm import tqdm


from utils.chunking import *
from utils.data import *
from utils.chromadb import *
from utils.visualization import *
from metrics import *
from embeddings import SentenceTransformerEmbeddingFunction
from chunking import FixedTokenChunker
from evaluation import Evaluation


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

CHUNKING_HYPERPARAMS = ['chunk_size', 'chunk_overlap', 'num_chunks']


def get_range(config: dict):
    return list(range(config['start'], config['end'], config['step'])) + [config['end']]


def get_pairs_of_params(config: dict):
    return list(filter(
        lambda x: x[1] <= x[0] / 2,
        product(*[
            get_range(config[name])
            for name in CHUNKING_HYPERPARAMS
        ])
    ))


class Experiment:

    def __init__(self, config: dict):
        self.CONFIG = config
        self.METRICS = config['metrics']

        self.PATHS = {name: Path(path) if path else None for name, path in config['paths'].items()}
        self.EXPERIMENT_PATHS = {}
        for partial_dir in ['questions', 'results']:
            path = self.PATHS['experiments'] / partial_dir
            path.mkdir(exist_ok=True, parents=True)
            self.EXPERIMENT_PATHS[partial_dir] = path

        self.evaluation = (
            Evaluation(
                questions_csv_path=self.PATHS['questions'],
                chroma_client=ChromaDBManager(self.PATHS['chroma']),
                corpora_path=self.PATHS['corpora'],
                embedding_function=SentenceTransformerEmbeddingFunction(**config['embeddings']),
                metrics=config['metrics'],
                batch_size=config['batch_size']
            )
        )

        self.use_wandb = self.PATHS['wandb'] is not None


    def explore_questions_df(self):
        logger.info('Exploring questions...')

        questions_df = load_questions_df(self.PATHS['questions'], self.PATHS['corpora'].stem)
        questions_df['n_references'] = questions_df['references'].apply(len).astype(int)

        create_histogram(
            questions_df, x='n_references',
            title='# References per Question',
            xlabel='# References/Question',
            ylabel='# Questions',
            discrete=True,
            shrink=0.1
        )
        plt.savefig(self.EXPERIMENT_PATHS['questions'] / 'num_refs_per_question_hist.png')
        if self.use_wandb:
            wandb.log({'questions/num_refs_per_question_hist': wandb.Image(plt)})
        plt.close()

        references_df = pd.DataFrame.from_records(
            questions_df['references'].explode().tolist()
        )

        references_df['length'] = references_df['content'].apply(len).astype(int)
        create_histogram(
            references_df, x='length',
            title='Reference Length',
            xlabel='Length',
            ylabel='# References',
            bins=20
        )
        plt.savefig(self.EXPERIMENT_PATHS['questions'] / 'reference_length_hist.png')
        if self.use_wandb:
            wandb.log({'questions/reference_length_hist': wandb.Image(plt)})
        plt.close()
        
        self.questions_statistics = {
            'corpora': self.PATHS['corpora'].stem,
            'n_questions': questions_df.shape[0],
            'n_references_per_question_mean': questions_df['n_references'].mean().item(),
            'n_references_per_question_std': questions_df['n_references'].std().item(),
            'n_references_total': questions_df['n_references'].sum().item(),
            'reference_length_mean': references_df['length'].mean().item(),
            'reference_length_std': references_df['length'].std().item()
        }
        with open(self.EXPERIMENT_PATHS['questions'] / 'statistics.json', "w") as f: 
            json.dump(self.questions_statistics, f)
        if self.use_wandb:
            self.wandb_run.config.update({"questions_statistics": self.questions_statistics})

        logger.info('Done.')

    def run_evaluation(self, chunk_size, chunk_overlap, num_chunks) -> dict:
        return (
            self.evaluation.run(
                chunker=FixedTokenChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                ),
                num_chunks=num_chunks
            )
        )

    def log_metrics_wandb(self, df: pd.DataFrame):
        df_copy = df.copy()
        for name in self.METRICS:
            df_copy[name] = (
                df_copy[f'{name}_mean'].map(lambda x: f"{x:.1f}") + 
                ' \u00B1 ' + 
                df_copy[f'{name}_std'].map(lambda x: f"{x:.1f}")
            )
            df_copy.drop(columns=[f'{name}_mean', f'{name}_std'], inplace=True)
        
        if self.use_wandb:
            wandb.log({'results/metrics_table': wandb.Table(dataframe=df_copy)})
    
    def run_evaluations(self):
        logger.info('Running evaluations...')

        metrics_list = []
        pba = tqdm(
            get_pairs_of_params(self.CONFIG),
            desc='Evaluation',
            leave=True, unit='configs'
        )
        for chunk_size, chunk_overlap, num_chunks in pba:
            metrics_list += [self.run_evaluation(chunk_size, chunk_overlap, num_chunks)]

        metrics_df = pd.DataFrame.from_records(metrics_list).sort_values(CHUNKING_HYPERPARAMS)

        metrics_df.to_csv(self.EXPERIMENT_PATHS['results'] / 'metrics.csv')
        if self.use_wandb:
            self.log_metrics_wandb(metrics_df)
            
        create_scatter(
            metrics_df, x='recall_mean', y='precision_mean',
            title='Recall vs Precision',
            xlabel='Recall',
            ylabel='Precision',
            trend=True
        )
        plt.savefig(self.EXPERIMENT_PATHS['results'] / 'recall_vs_precision.png')
        if self.use_wandb:
            wandb.log({'results/recall_vs_precision': plt})
        plt.close()

        logger.info('Done.')

        return metrics_df

    def create_heatmaps(self, df: pd.DataFrame):
        logger.info('Creating heatmaps...')
        modes = ['original', 'interpolated']
        
        for mode in modes:
            (self.EXPERIMENT_PATHS['results'] / 'heatmaps' / mode).mkdir(parents=True, exist_ok=True)
        
        combinations = [(v, [x for x in CHUNKING_HYPERPARAMS if x != v]) for v in CHUNKING_HYPERPARAMS]
        vmaxes = {metric: df[f"{metric}_mean"].max() for metric in self.METRICS}
        
        for (group_by, [x, y]), mode in product(combinations, modes):
            for value in df[group_by].unique().tolist():
                create_heatmap(
                    df, self.METRICS, vmaxes, 
                    group_by, value, x, y, 
                    interpolated=(mode == 'interpolated')
                )
                name = f'{x}_vs_{y}_{group_by}_{value}'
                
                plt.savefig(self.EXPERIMENT_PATHS['results'] / 'heatmaps' / mode / f'{name}.png')
                # if self.use_wandb:
                #     wandb.log({f'results/heatmaps/{mode}/{name}': wandb.Image(plt)})
                plt.close()
        
        logger.info('Done.')


    def create_plots(self, df: pd.DataFrame):
        logger.info('Creating plots...')
        (self.EXPERIMENT_PATHS['results'] / 'trends').mkdir(exist_ok=True)
        
        for param in CHUNKING_HYPERPARAMS:
            create_plot(df, self.METRICS, param)
            plt.savefig(self.EXPERIMENT_PATHS['results'] / 'trends' / f'{param}.png')
            if self.use_wandb:
                wandb.log({f'results/trends/{param}': wandb.Image(plt)})
            plt.close()
        
        logger.info('Done.')
            

    def run(self):
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project="chunking_evaluation", 
                name="fixed_token_chunker", 
                dir=self.PATHS['wandb'], 
                job_type="eval"
            )

        self.explore_questions_df()
        metrics_df = self.run_evaluations()
        self.create_heatmaps(metrics_df)
        self.create_plots(metrics_df)

        if self.use_wandb:
            self.wandb_run.finish()


def read_config():
    with open(Path(__file__).parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = read_config()
    Experiment(config).run()


if __name__ == "__main__":
    main()
