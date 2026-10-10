#!/bin/zsh
# Install the optional FaceTime voice add-on on this Mac (run as the user who is logged in to FaceTime).
#
#   catanbot/facetime/macos/install.sh            dry run: says what it would do and what is missing
#   catanbot/facetime/macos/install.sh --apply    pip install requirements-facetime.txt into the repo's
#                                                 .venv, build facetime-nc and the CatanbotFaceTime.app
#                                                 launcher into ~/.catanbot-facetime/, install the
#                                                 LaunchAgent local.catanbot.facetime, (re)start it
#
# Prerequisites it never does for you (docs/facetime.md): BlackHole 2ch + 16ch, FaceTime's speaker and
# microphone set to them, Remote Login (ssh localhost with a key) + Accessibility for sshd, the
# Microphone permission for CatanbotFaceTime.app (one Allow click), a "facetime" block in data/config.json.
# Never touches coreaudiod or any other LaunchAgent.
set -euo pipefail

HERE="${0:A:h}"
REPO="${HERE:h:h:h}"
DEST="$HOME/.catanbot-facetime"
LABEL="local.catanbot.facetime"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
APP="$DEST/CatanbotFaceTime.app"
PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/scripts/facetime_voice.py"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

say() { print -r -- "install: $*"; }
do_() { if (( APPLY )); then "$@"; else say "would run: $*"; fi; }

reload_agent() {
  local dom="gui/$(id -u)" i
  launchctl bootout "$dom/$LABEL" 2>/dev/null || true
  for i in {1..20}; do
    launchctl print "$dom/$LABEL" >/dev/null 2>&1 || break
    sleep 0.5
  done
  launchctl enable "$dom/$LABEL" 2>/dev/null || true
  for i in {1..4}; do
    launchctl bootstrap "$dom" "$PLIST" && break
    say "bootstrap attempt $i failed; retrying"
    sleep 2
    launchctl enable "$dom/$LABEL" 2>/dev/null || true
  done
  if launchctl print "$dom/$LABEL" >/dev/null 2>&1; then
    say "loaded $LABEL"
    return 0
  fi
  print -u2 -r -- "
!! $LABEL IS NOT LOADED. Run, as $(id -un), in a Terminal on this Mac's console:
!!   launchctl enable $dom/$LABEL
!!   launchctl bootstrap $dom $PLIST
!! Check: launchctl print $dom/$LABEL | grep state"
  exit 1
}

# The bundle is rebuilt only when launcher.c, Info.plist or the baked paths change: the microphone
# grant is tied to its ad-hoc signature, so a rebuild needs the owner's Allow click again.
build_app() {
  local want have
  want=$( { cat "$HERE/app/launcher.c" "$HERE/app/Info.plist"; print -r -- "$PY $SCRIPT"; } | shasum -a 256 | cut -c1-16)
  have=$(cat "$APP/Contents/Resources/source.sha" 2>/dev/null || true)
  if [[ "$want" == "$have" ]] && codesign --verify "$APP" 2>/dev/null; then
    say "CatanbotFaceTime.app unchanged ($want)"
    return 0
  fi
  say "building CatanbotFaceTime.app ($want); click Allow again for the microphone (docs/facetime.md)"
  rm -rf "$APP.tmp"
  mkdir -p "$APP.tmp/Contents/MacOS" "$APP.tmp/Contents/Resources"
  cp "$HERE/app/Info.plist" "$APP.tmp/Contents/Info.plist"
  xcrun cc -Wall -O2 -DCATANBOT_PY="\"$PY\"" -DCATANBOT_SCRIPT="\"$SCRIPT\"" \
    -o "$APP.tmp/Contents/MacOS/catanbot-facetime" "$HERE/app/launcher.c"
  print -r -- "$want" > "$APP.tmp/Contents/Resources/source.sha"
  codesign --force --sign - --identifier "$LABEL" "$APP.tmp"
  rm -rf "$APP" && mv "$APP.tmp" "$APP"
  codesign --verify --verbose=1 "$APP"
}

# facetime-nc: the native-AX 1:1 ring helper. It runs under `ssh localhost` like facetime-ax.js,
# so sshd's Accessibility grant covers it and a rebuild needs no new grant.
build_nc() {
  /usr/bin/swiftc -O -o "$DEST/facetime-nc.tmp" "$HERE/facetime-nc.swift"
  mv "$DEST/facetime-nc.tmp" "$DEST/facetime-nc"
  say "built $DEST/facetime-nc"
}

missing=0
[[ -x $PY ]] || { say "no $PY (create the repo venv first: README)"; exit 1; }
[[ -x /usr/bin/swiftc ]] || { say "no /usr/bin/swiftc (xcode-select --install) for facetime-nc"; exit 1; }
xcrun --find cc >/dev/null 2>&1 || { say "no C compiler (xcode-select --install) for the app launcher"; exit 1; }
if ! system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole 16ch"; then say "MISSING: BlackHole 16ch"; missing=1; fi
if ! system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole 2ch"; then say "MISSING: BlackHole 2ch"; missing=1; fi
"$PY" "$SCRIPT" --check 2>/dev/null | grep -q '"enabled": true' || { say "MISSING: a \"facetime\" block in $REPO/data/config.json"; missing=1; }
ssh -o BatchMode=yes -o ConnectTimeout=5 -T localhost true 2>/dev/null || { say "MISSING: ssh localhost (BatchMode) for the AX helpers (or set tuning.ax_via to \"direct\")"; missing=1; }
if (( APPLY && missing )); then say "fix the MISSING lines first"; exit 1; fi

do_ mkdir -p "$DEST" "$HOME/Library/LaunchAgents"
do_ "$PY" -m pip install --quiet -r "$REPO/requirements-facetime.txt"
if (( APPLY )); then
  (cd "$REPO" && PYTHONPATH=. "$PY" -m pytest -q tests/test_facetime*.py 2>&1 | tail -n 1)
  build_nc
  build_app
  sed "s#__DEST__#$DEST#g" "$HERE/$LABEL.plist" > "$PLIST.tmp" && mv "$PLIST.tmp" "$PLIST"
  plutil -lint "$PLIST" >/dev/null
  reload_agent
  sleep 3
  launchctl print "gui/$(id -u)/$LABEL" | grep -E '^\s*(state|pid|last exit code)' || true
  tail -n 5 "$DEST/facetime.log" 2>/dev/null || true
  say "microphone: once, from the GUI session: open -n -W \"$APP\" --args --mic-check ; then: tail -1 $DEST/mic-check.log"
else
  say "would build $DEST/facetime-nc (swiftc facetime-nc.swift)"
  say "would build $APP (launcher.c + Info.plist, ad-hoc signed) if they changed"
  say "would write $PLIST and launchctl bootout + enable + bootstrap gui/$(id -u)/$LABEL"
  say "dry run only; re-run with --apply"
fi
