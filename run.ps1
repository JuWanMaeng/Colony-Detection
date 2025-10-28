# run.ps1
Write-Host "🚀 Colony environment training started"
conda activate colony
python train_v12_small.py
python train_v12_nano.py
python train_v8_small.py

Write-Host "✅ All done!"
