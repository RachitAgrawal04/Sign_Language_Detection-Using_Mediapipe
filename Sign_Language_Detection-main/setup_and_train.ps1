# Sign Language Interpreter - Automated Setup Script
# PowerShell script for easy training workflow

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Sign Language Interpreter - Training Pipeline  " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if gestures folder exists
if (-Not (Test-Path "gestures")) {
    Write-Host "❌ No gestures folder found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "First, capture gestures by running:" -ForegroundColor Yellow
    Write-Host "  python Code/create_gestures_mediapipe.py" -ForegroundColor Green
    Write-Host ""
    Write-Host "Capture at least 6 different gestures (IDs: 1, 2, 3, 4, 5, 6)" -ForegroundColor Yellow
    exit 1
}

# Count gestures
$gestureCount = (Get-ChildItem "gestures" -Directory).Count
Write-Host "✓ Found $gestureCount gesture folders" -ForegroundColor Green

if ($gestureCount -lt 2) {
    Write-Host "❌ Need at least 2 gestures to train!" -ForegroundColor Red
    Write-Host "Run: python Code/create_gestures_mediapipe.py" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Starting training pipeline..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Split dataset
Write-Host "Step 1/2: Splitting dataset..." -ForegroundColor Yellow
python Code/load_images.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Dataset splitting failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dataset split successfully" -ForegroundColor Green
Write-Host ""

# Step 2: Train model
Write-Host "Step 2/2: Training model (this may take 5-15 minutes)..." -ForegroundColor Yellow
python Code/cnn_model_train.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Model training failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Model trained successfully" -ForegroundColor Green
Write-Host ""

# Success
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  ✅ Training Complete!                          " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model saved to: cnn_model_keras2.h5" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next step: Test your model!" -ForegroundColor Cyan
Write-Host "  python Code/final_mediapipe.py" -ForegroundColor Green
Write-Host ""
