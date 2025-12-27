import argparse
import os
import shutil
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast
from google.cloud import storage

# Importamos tus utilidades
import distilbert_utils as distilbert_utils

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Descarga un archivo desde GCS."""
    print(f"Descargando gs://{bucket_name}/{source_blob_name} -> {destination_file_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def upload_directory(local_path, bucket_name, gcs_path):
    """Sube todo el contenido de un directorio a GCS."""
    print(f"Subiendo resultados a gs://{bucket_name}/{gcs_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            # Calcular ruta relativa para mantener estructura
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)

def main(args):
    # Preparar Entorno Local (Dentro del contenedor)
    # Limpiamos si existen para evitar residuos
    if os.path.exists("data"): shutil.rmtree("data")
    if os.path.exists("results"): shutil.rmtree("results")
    
    os.makedirs("data", exist_ok=True)
    out_dir = "results"
    label_dir = os.path.join(out_dir, "labels")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Descargar Datos
    if args.data_path.startswith("gs://"):
        # Parsear gs://bucket/path
        parts = args.data_path.replace("gs://", "").split("/")
        bucket_name = parts[0]
        blob_name = "/".join(parts[1:])
        local_data_path = os.path.join("data", "dataset.txt")
        download_blob(bucket_name, blob_name, local_data_path)
    else:
        # Modo local para pruebas
        local_data_path = args.data_path

    # Cargar y Preprocesar Datos
    print(f"=== Cargando datos desde {local_data_path} ===")
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
    
    # Debugging: Sampleo opcional
    if args.sample_frac < 1.0:
        print(f"AVISO: Usando submuestra del {args.sample_frac*100}%")
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)

    print(f"Dataset final: {df.shape} filas.")

    # Configurar Estrategia de Entrenamiento
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

    print(f"=== Configuración: {args.train_type.upper()} ===")
    print(f"Fine Tune: {fine_tune}, Layers: {n_finetune_layers}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Ejecutar Entrenamiento Iterativo
    # Llamamos a tu función utils.iterative_training
    distilbert_utils.iterative_training(
        train_type=args.train_type,
        text_col='GOODS_DESCRIPTION',   # Fijo
        target_col='HS04',              # Fijo
        iterations=args.iterations,
        max_epochs=args.max_epochs,
        max_length=args.max_length,
        loader_batch_size=args.batch_size,
        shuffle=True,
        lr=args.lr,
        fraction=args.test_fraction,
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

    # Guardar Resultados en GCS
    # Subimos a la carpeta especificada en args.job_dir
    upload_directory(out_dir, args.output_bucket, args.job_dir)
    print("=== Trabajo Terminado Exitosamente ===")

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

    args = parser.parse_args()
    main(args)