#!/usr/bin/env bash
# Fetch the Azure 2019 VM trace (CC BY 4.0) used by E11.
# 437 MB; not committed.  See data/README.md for the schema and the checksum.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/azure2019
BASE=https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2

for f in schema.csv azure2019_data_category.txt azure2019_data_cores.txt \
         azure2019_data_lifetime.txt azure2019_data_memory.txt \
         azure2019_data_cpu.txt azure2019_data_deployment.txt; do
  [ -f "data/azure2019/$f" ] || curl -sSL -o "data/azure2019/$f" "$BASE/$f"
done

if [ ! -f data/azure2019/vmtable.csv.gz ]; then
  echo "fetching vmtable.csv.gz (437 MB) ..."
  curl -L -o data/azure2019/vmtable.csv.gz \
    "$BASE/trace_data_vmtable_vmtable.csv.gz"
fi

echo "expected sha256 e8c9a0ab0e06b4322747147b49f8b44c07cf017327a150a92b6fb95aab8de3e5"
sha256sum data/azure2019/vmtable.csv.gz 2>/dev/null || \
  shasum -a 256 data/azure2019/vmtable.csv.gz
echo "now run: python experiments/derive_azure_workload.py"
