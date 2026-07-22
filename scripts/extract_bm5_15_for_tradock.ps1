# Extract BM5_15complexes.zip and copy labels for TraDock eval.
$ErrorActionPreference = 'Stop'
$zip = 'C:\Users\yang.nan\Desktop\BM5_15complexes.zip'
$destRoot = 'C:\Users\yang.nan\Desktop\score\tradock_data'
$labelsSrc = 'C:\Users\yang.nan\Desktop\AUC&ClassificanMetrics&SuccessRate\AUC&ClassificanMetrics&SuccessRate\BM5_scores&labels.csv'

New-Item -ItemType Directory -Force -Path $destRoot | Out-Null
Write-Host "Extracting $zip -> $destRoot ..."
Expand-Archive -Path $zip -DestinationPath $destRoot -Force
Copy-Item -Force $labelsSrc (Join-Path $destRoot 'BM5_scores&labels.csv')
$pdbCount = (Get-ChildItem (Join-Path $destRoot 'BM5_15complexes\PDBs') -Filter *.pdb).Count
Write-Host "PDBs=$pdbCount"
Write-Host "Ready:"
Write-Host "  data:   $(Join-Path $destRoot 'BM5_15complexes')"
Write-Host "  labels: $(Join-Path $destRoot 'BM5_scores&labels.csv')"
Write-Host "Upload both to the GPU server, then run scripts/run_bm5_15_tradock_eval.sh"
