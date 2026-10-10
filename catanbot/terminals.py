"""Owner 2026-10-04: the bot must not leave dozens of Terminal windows open; close them after
every game. Every bot launch (start_bot.command and the like) opens a Terminal window that stays
on "[Process completed]" once the bot exits. Close only idle windows whose title names a bot
launcher; never touch a busy window (the running bot, a Claude session) or any other shell."""
import subprocess
import sys

BOT_TITLES = ("start_bot.command", "runner_only.command", "probe.command", "snap.command",
              "fit_setup.command", "train.command", "marlow-bot")

SCRIPT = """
set n to 0
tell application "Terminal"
  set ids to {}
  repeat with w in (every window)
    try
      set t to name of w
      if (busy of every tab of w) does not contain true then
        repeat with k in {%s}
          if t contains (k as text) then
            set end of ids to id of w
            exit repeat
          end if
        end repeat
      end if
    end try
  end repeat
  -- close by id after the scan: closing while iterating shifts the window indices
  repeat with i in ids
    try
      close (first window whose id is i) saving no
      set n to n + 1
    end try
  end repeat
end tell
return n
"""


def applescript(titles=BOT_TITLES) -> str:
    return SCRIPT % ", ".join('"%s"' % t for t in titles)


def close_idle_bot_terminals(run=subprocess.run) -> int:
    """Close idle bot-launcher Terminal windows; return how many were closed (0 on any error)."""
    if sys.platform != "darwin":
        return 0
    try:
        r = run(["osascript", "-e", applescript()], capture_output=True, text=True, timeout=20)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return 0


if __name__ == "__main__":
    print(close_idle_bot_terminals())
