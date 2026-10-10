#!/bin/zsh
cd "$(dirname "$0")"
.venv/bin/python -c "
from catanbot import mac
print('before:', mac.permissions_ok())
mac.request_permissions()
try:
    img, w = mac.capture(); print('capture OK', img.shape)
except Exception as e: print('capture:', e)
print('after:', mac.permissions_ok())
"
echo "If anything says False: System Settings > Privacy & Security > Screen Recording / Accessibility > enable Terminal, then re-run."
