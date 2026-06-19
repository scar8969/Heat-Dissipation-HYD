"""
Urban Heat Mitigation - Credential Setup Script

Run this script once to authenticate with all required services.
Usage: python setup_credentials.py
"""

import os
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "urban-heat"


def setup_gee():
    """Authenticate with Google Earth Engine."""
    print("\n[1/6] Google Earth Engine")
    print("  1. Go to https://earthengine.google.com/")
    print("  2. Sign up for a research/education account")
    print("  3. Run: earthengine authenticate")
    try:
        import ee
        ee.Initialize()
        print("  ✓ GEE authenticated successfully")
    except Exception:
        print("  ⚠ Run 'earthengine authenticate' to complete setup")


def setup_cds_api():
    """Create CDS API configuration."""
    print("\n[2/6] CDS API (ERA5-Land)")
    print("  1. Register at https://cds.climate.copernicus.eu/")
    print("  2. Go to https://cds.climate.copernicus.eu/api/v2")
    print("  3. Copy your UID and API key")

    cds_config = Path.home() / ".cdsapirc"
    if cds_config.exists():
        print(f"  ⚠ {cds_config} already exists. Skipping.")
        return

    uid = input("  Enter CDS UID (or press Enter to skip): ").strip()
    if not uid:
        print("  ⚠ Skipped. Create ~/.cdsapirc manually.")
        return

    api_key = input("  Enter CDS API Key: ").strip()
    content = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: {uid}:{api_key}\n"
    cds_config.write_text(content)
    print(f"  ✓ CDS API config written to {cds_config}")


def setup_cpcb():
    """Setup CPCB air quality data access."""
    print("\n[3/6] CPCB Air Quality")
    print("  1. Visit https://cpcb.nic.in/")
    print("  2. Request API access for Hyderabad station data")
    print("  3. Alternatively, use the open data portal")
    print("  Note: This is optional. ERA5 can serve as proxy.")


def setup_imd():
    """Setup IMD weather station access."""
    print("\n[4/6] IMD Weather Stations")
    print("  1. Visit https://imdpune.gov.in/ or https://mosdac.gov.in/")
    print("  2. Register for station data access")
    print("  Hyderabad stations: Begumpet, Shamshabad airport")
    print("  Note: This is optional. ERA5-Land can serve as proxy.")


def setup_wandb():
    """Setup Weights & Biases for experiment tracking."""
    print("\n[5/6] Weights & Biases (Optional)")
    print("  Run: wandb login")
    try:
        import wandb
        wandb.login()
        print("  ✓ W&B authenticated")
    except Exception:
        print("  ⚠ Run 'wandb login' to complete setup")


def setup_huggingface():
    """Setup Hugging Face for model weights."""
    print("\n[6/6] Hugging Face (Optional)")
    print("  For US-UrbanLST dataset and Earthformer weights")
    print("  Run: huggingface-cli login")


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    dirs = [
        CONFIG_DIR,
        Path("data/raw"),
        Path("data/processed"),
        Path("data/splits"),
        Path("outputs/models"),
        Path("outputs/maps"),
        Path("outputs/rasters"),
        Path("outputs/optimization"),
        Path("outputs/dashboard"),
        Path("outputs/reports"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")


def main():
    print("=" * 60)
    print("  Urban Heat Mitigation - Credential Setup")
    print("=" * 60)

    create_directories()

    setup_gee()
    setup_cds_api()
    setup_cpcb()
    setup_imd()
    setup_wandb()
    setup_huggingface()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Start with notebook: notebooks/01_gee_acquisition.ipynb")


if __name__ == "__main__":
    main()
