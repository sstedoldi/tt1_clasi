# GCP Task Script for DistilBERT Training
# gcp/gcp_task.py
import argparse
import os
import shutil
import pandas as pd
import logging
from transformers import DistilBertTokenizerFast
from google.cloud import storage

# Local utils
import distilbert_utils as distilbert_utils

def download_blob(bucket_name, source_blob_name, destination_file_name, logger=logging.getLogger(__name__)):
    """Descarga un archivo desde GCS."""
    logger.info(f"Descargando gs://{bucket_name}/{source_blob_name} -> {destination_file_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def upload_directory(local_path, bucket_name, gcs_path, logger):
    """Sube todo el contenido de un directorio a GCS."""
    logger.info(f"Subiendo resultados a gs://{bucket_name}/{gcs_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            # Relative path
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)

def setup_logging(out_dir):
    log_file = os.path.join(out_dir, "training_log.log")
    
    # Root logger config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file), # Save to file
            logging.StreamHandler()        # Also print to console
        ]
    )
    return log_file

def main(args):
    # Local files preparation (inside container)
    if os.path.exists("data"): shutil.rmtree("data")
    if os.path.exists("results"): shutil.rmtree("results")

    os.makedirs("data", exist_ok=True)
    out_dir = "results"
    label_dir = os.path.join(out_dir, "labels")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Logging cofig
    log_file_path = os.path.join(out_dir, "training_execution.log")
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()          
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Iniciando Tarea en GCP ===")
    logger.info(f"Archivo de log local: {log_file_path}")

    #  Data Loading
    if args.data_path.startswith("gs://"):
        # Parsing gs://bucket/path
        parts = args.data_path.replace("gs://", "").split("/")
        bucket_name = parts[0]
        blob_name = "/".join(parts[1:])
        local_data_path = os.path.join("data", "dataset.txt")
        download_blob(bucket_name, blob_name, local_data_path, logger)
    else:
        # Local mode
        local_data_path = args.data_path

    # Loading and processing data
    logger.info(f"=== Cargando datos desde {local_data_path} ===")
    colspecs = [(0, 6), (6, None)]
    df = pd.read_fwf(
        local_data_path,
        colspecs=colspecs, 
        header=None,
        names=['HS06', 'GOODS_DESCRIPTION'],
        dtype={'HS06': str}
    )
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Feature Engineering (HS04)
    df['HS04'] = df['HS06'].str[:4]
    
    # Debugging: optional sub-sampling
    if args.sample_frac < 1.0:
        logger.info(f"AVISO: Usando submuestra del {args.sample_frac*100}%")
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)

    logger.info(f"Dataset final: {df.shape} filas.")

    # Training setup config
    # FE (Fixed Encoder): fine_tune=False, layers=0
    # FFT (Full Fine-Tune): fine_tune=True, layers=0 (0 implica todas en tu utils)
    # PFT (Partial Fine-Tune): fine_tune=True, layers=2
    
    if args.train_type == 'fe':
        fine_tune = False
        n_finetune_layers = 0
    elif args.train_type == 'fft':
        fine_tune = True
        n_finetune_layers = 0 
    elif args.train_type == 'pft':
        fine_tune = True
        n_finetune_layers = 2
    else:
        raise ValueError("Tipo de entrenamiento desconocido")

    logger.info(f"=== Configuración: {args.train_type.upper()} ===")
    logger.info(f"Fine Tune: {fine_tune}, Layers: {n_finetune_layers}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    logger.info("=== Iniciando Entrenamiento ===")

    if args.final:
        logger.info("=== MODO FINAL: single run ===")
        distilbert_utils.training(
            train_type=args.train_type,
            text_col="GOODS_DESCRIPTION",
            target_col="HS04",
            df=df,
            tokenizer=tokenizer,
            out_dir=out_dir,
            label_dir=label_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            lr=args.lr,
            test_fraction=args.test_fraction,  # set to 0.01 in job
            seed=args.seed,
            max_epochs=args.max_epochs,
            early_stopping=True,
            monitor="val_loss",
            patience=3,
            warmup_epochs=1,
            fine_tune=fine_tune,
            n_finetune_layers=n_finetune_layers,
            num_workers=args.num_workers,
            verbose=True,
        )
    else:
        logger.info("=== MODO ITERATIVO ===")
        distilbert_utils.iterative_training(
            train_type=args.train_type,
            text_col='GOODS_DESCRIPTION',
            target_col='HS04',
            iterations=args.iterations,
            max_epochs=args.max_epochs,
            max_length=args.max_length,
            loader_batch_size=args.batch_size,
            shuffle=True,
            lr=args.lr,
            fraction=args.test_fraction,
            bootstrap=args.bootstrap,
            out_dir=out_dir,
            df=df,
            tokenizer=tokenizer,
            label_dir=label_dir,
            fine_tune=fine_tune,
            n_finetune_layers=n_finetune_layers,
            patience=3,
            monitor="val_loss",
            num_workers=args.num_workers,
            verbose=True
        )

    # Saving results GCS into args.job_dir
    upload_directory(out_dir, args.output_bucket, args.job_dir, logger)
    logger.info("=== Trabajo Terminado Exitosamente ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GCP Args
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_bucket', type=str, required=True)
    parser.add_argument('--job_dir', type=str, required=True) # Carpeta destino en bucket
    
    # Training Args
    parser.add_argument('--train_type', type=str, required=True, choices=['fe', 'fft', 'pft'])
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--test_fraction', type=float, default=0.05) # 5% test
    parser.add_argument('--sample_frac', type=float, default=1.0)   # 1.0 = 100% data
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument(
        '--bootstrap', 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="Usa --bootstrap para activar o --no-bootstrap para desactivar"
        )
    parser.add_argument(
    "--final",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use --final for a single definitive training run (no iterations)."
    )
    
    args = parser.parse_args()
    main(args)