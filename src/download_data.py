"""Télécharge le dataset Telco Customer Churn (IBM Sample) dans data/."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "Telco-Customer-Churn.csv"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, OUT)
    print(f"Fichier téléchargé : {OUT} ({OUT.stat().st_size} octets)")

if __name__ == "__main__":
    main()