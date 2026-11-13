#!/bin/bash
# =================================================================
# EC2 Server Setup Script — Heart Disease Prediction App
# =================================================================
# Run this ONCE after launching a fresh AWS EC2 Ubuntu/Amazon Linux instance
# 
# Usage: bash scripts/setup_ec2.sh
# =================================================================

set -e  # Exit immediately if any command fails

echo "================================================="
echo "  EC2 Setup — Heart Disease Prediction App"
echo "================================================="

# ── Step 1: System Update ──────────────────────────────────────
echo ""
echo "[1] Updating system packages..."
sudo yum update -y 2>/dev/null || sudo apt-get update -y
echo "    ✓ System updated"

# ── Step 2: Install Python 3 ──────────────────────────────────
echo ""
echo "[2] Installing Python 3..."
sudo yum install -y python3 python3-pip python3-venv 2>/dev/null || \
  sudo apt-get install -y python3 python3-pip python3-venv
echo "    ✓ Python installed: $(python3 --version)"

# ── Step 3: Clone or update the application ───────────────────
echo ""
echo "[3] Setting up application directory..."
APP_DIR="/home/ec2-user/heart-disease-app"

if [ -d "$APP_DIR" ]; then
  echo "    App directory already exists. Pulling latest..."
  cd "$APP_DIR"
else
  echo "    Creating app directory..."
  mkdir -p "$APP_DIR"
  cd "$APP_DIR"
fi

# ── Step 4: Python virtual environment ────────────────────────
echo ""
echo "[4] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "    ✓ Virtual environment created"
else
  echo "    ✓ Virtual environment already exists"
fi

source venv/bin/activate
pip install --upgrade pip --quiet

# ── Step 5: Install requirements ──────────────────────────────
echo ""
echo "[5] Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "    ✓ Dependencies installed"

# ── Step 6: Train the model ───────────────────────────────────
echo ""
echo "[6] Training the Random Forest model..."
python scripts/train_model.py
echo "    ✓ Model trained and saved"

# ── Step 7: Install systemd service ───────────────────────────
echo ""
echo "[7] Installing systemd service..."
sudo cp scripts/heart-disease-app.service /etc/systemd/system/heart-disease-app.service
sudo systemctl daemon-reload
sudo systemctl enable heart-disease-app
sudo systemctl start heart-disease-app
echo "    ✓ Systemd service installed and started"

# ── Step 8: Verify ────────────────────────────────────────────
echo ""
echo "[8] Verifying service status..."
sleep 3
if sudo systemctl is-active --quiet heart-disease-app; then
  echo "    ✓ Application is RUNNING on port 8501"
else
  echo "    ✗ Application failed to start. Check logs:"
  echo "      sudo journalctl -u heart-disease-app -n 50"
  exit 1
fi

# ── Get public IP ─────────────────────────────────────────────
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")

echo ""
echo "================================================="
echo "  ✓ SETUP COMPLETE"
echo "================================================="
echo ""
echo "  App URL   : http://${PUBLIC_IP}:8501"
echo "  Logs      : sudo journalctl -u heart-disease-app -f"
echo "  Restart   : sudo systemctl restart heart-disease-app"
echo "  Status    : sudo systemctl status heart-disease-app"
echo ""
echo "  IMPORTANT: Open port 8501 in your EC2 Security Group!"
echo "================================================="
