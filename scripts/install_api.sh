#!/bin/zsh
# Install the read-only Catan API as a LaunchAgent for the current user (macOS).
# Run from the repo:  zsh scripts/install_api.sh   (LABEL and KEY can be set in the environment)
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
KEY=${KEY:-~/.catanbot-api.key}
if [ ! -f "$KEY" ]; then
  umask 077
  .venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))" > "$KEY"
fi
chmod 600 "$KEY"
LABEL=${LABEL:-local.catanbot.api}
PLIST=~/Library/LaunchAgents/$LABEL.plist
mkdir -p ~/Library/LaunchAgents
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$REPO/.venv/bin/python</string><string>-m</string><string>catanbot.api</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>$REPO</string><key>CATANBOT_API_KEY_FILE</key><string>$KEY</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>Nice</key><integer>5</integer>
  <key>StandardErrorPath</key><string>$REPO/data/api.log</string>
  <key>StandardOutPath</key><string>$REPO/data/api.log</string>
</dict></plist>
EOF
launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$PLIST"
sleep 2
curl -s http://127.0.0.1:8940/health; echo
echo "token last 4: $(tail -c 5 $KEY)"
