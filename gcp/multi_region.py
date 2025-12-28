import argparse
import subprocess
import sys
import time

REGIONS_PRIORITY = [
    # using only zones with g2-standard-4
    "us-central1",      # Iowa
    "us-east1",         # South Carolina
    "us-west1",         # Oregon
    # "us-west4",         # Las Vegas
    "us-east4",         # Virginia
    # "asia-east1",       # Taiwan
    # "asia-northeast1",  # Tokyo
    "asia-southeast1",  # Singapore
    "europe-west4",   # Netherlands
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Launcher Multi-Región para Vertex AI")
    parser.add_argument("--display-name", required=True, help="Nombre visible del Job en GCP")
    parser.add_argument("--config", required=True, help="Ruta al archivo .yaml o .json de configuración")
    return parser.parse_args()

def run_job(region, display_name, config_file):
    """
    Intenta lanzar el job en una región.
    Retorna True si el job fue aceptado (Exit code 0).
    Retorna False si falló.
    """
    print(f"\n🌍 Intentando región: \033[1m{region}\033[0m ...")
    
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--region={region}",
        f"--display-name={display_name}",
        f"--config={config_file}",
        "--format=json"
    ]

    is_windows = sys.platform.startswith('win')

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=is_windows
        )
        
        print(f"✅ \033[92mÉXITO: Job lanzado en {region}\033[0m")
        print("📝 Output de GCP:")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.lower()
        
        if "quota" in error_msg or "resource exhausted" in error_msg or "not found" in error_msg:
            print(f"⚠️  Fallo de Recursos/Stock en {region}. Pasando a la siguiente...")
        else:
            print(f"❌ Error CRÍTICO en {region} (No parece ser de stock):")
            print(e.stderr)
            if "invalid" in error_msg or "argument" in error_msg:
                print("🛑 Deteniendo script por error de configuración.")
                sys.exit(1)
                
        return False

def main():
    args = parse_arguments()
    
    print(f"🚀 Iniciando Launcher para: {args.display_name}")
    print(f"📄 Config: {args.config}")
    print("------------------------------------------------")

    job_launched = False
    while not job_launched:
        for region in REGIONS_PRIORITY:
            success = run_job(region, args.display_name, args.config)
            if success:
                job_launched = True
                print("------------------------------------------------")
                print(f"🎉 Proceso finalizado. El job está corriendo en {region}.")
                print("🛑 Deteniendo script para evitar costos duplicados.")
                break
            time.sleep(3)

        if not job_launched:
            print("\n💀 No se pudo lanzar el job en ninguna de las regiones listadas.")
            print("💡 Sugerencia: Verifica si hay problemas de cuota o stock en las regiones.")
            print("♻️  Reintentando desde el inicio de la lista de regiones...")
            # sys.exit(1)

if __name__ == "__main__":
    main()