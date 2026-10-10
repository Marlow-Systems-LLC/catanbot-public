# catanbot

A bot that plays **Catan Universe** (Steam) on macOS. It focuses on the
**Cities & Knights: The Great Canal** scenario and also handles base Catan and Seafarers.

It reads the screen with OpenCV and Apple Vision OCR, works out the board, hands and prompts,
picks a move with a rule-and-weight "brain", and clicks through the game's own UI. A built-in
simulator runs the same brain in self-play, so it can be trained and A/B tested offline without
the game running.

> Unofficial fan project. Not affiliated with or endorsed by CATAN GmbH, Asmodee or the makers of
> Catan Universe. "Catan" is a trademark of CATAN GmbH. Use it in **private games with friends or
> against the computer only**. The bot never presses public matchmaking buttons
> (`never_matchmaking` in the config).

## Requirements

- macOS 13 or later (it uses Quartz for screen capture and input, and Apple Vision for OCR)
- Python 3.11
- Catan Universe from Steam, for live play only. The simulator and tests don't need it.

## Quick start with Docker

Docker runs everything except live play: tests, simulation, training and the API. Live play needs
macOS (see below). You need Docker Desktop or any Docker engine with Compose.

```bash
git clone https://github.com/Marlow-Systems-LLC/catanbot-public.git
```

```bash
cd catanbot-public
```

Run the tests (the macOS-only ones are skipped in the container):

```bash
docker compose run --rm tests
```

Run any script in `scripts/`. Models and results are written to `./data` on your machine.

```bash
docker compose run --rm sim scripts/selfplay.py 100
```

```bash
docker compose run --rm sim scripts/battle.py 200 data/model.json data/model.json great_canal
```

```bash
docker compose run --rm sim scripts/train_loop.py 2 great_canal
```

Start the read-only API on `http://localhost:8940`. Make a token first.

```bash
mkdir -p secrets && python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/api.key
```

```bash
docker compose up -d api
```

```bash
curl -H "Authorization: Bearer $(cat secrets/api.key)" http://localhost:8940/version
```

## Native install (macOS, needed for live play)

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Optional: turn on the pre-commit hook that rebuilds the strategy RAG whenever its docs change.

```bash
git config core.hooksPath .githooks
```

### macOS permissions (live play only)

The bot captures the screen and sends clicks, so the app you start it from (Terminal, iTerm, …)
needs **Screen Recording** and **Accessibility** in System Settings → Privacy & Security. Check
them with:

```bash
zsh check_permissions.command
```

Anything that prints `False` still needs the permission. Grant it, then run the check again.

## Run the tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

The tests run on recorded frames and fixtures. They don't need the game or any permissions.

## Simulate and train (no game needed)

Self-play with the same brain in all four seats:

```bash
PYTHONPATH=. .venv/bin/python scripts/selfplay.py 100
```

Two models play each other with two seats each, and the seating alternates between games:

```bash
PYTHONPATH=. .venv/bin/python scripts/battle.py 200 data/model.json data/model.json great_canal
```

A/B test a single brain rule (switching it off against the current model):

```bash
PYTHONPATH=. .venv/bin/python scripts/ab_policy.py 400 <rule> great_canal data/model.json
```

Training loop: each round trains a candidate model and keeps it only if it beats the current best.
The argument is the number of hours to run.

```bash
PYTHONPATH=. .venv/bin/python scripts/train_loop.py 2 great_canal
```

## Live play

1. Start Catan Universe in full screen and leave it as the front window.
2. Copy the settings you want to change into `data/config.json` (see below).
3. Start the runner:

```bash
PYTHONPATH=. .venv/bin/python -m catanbot.runner
```

- `--dry-run` reads the screen and decides moves but never clicks.
- `--replay FRAME.png …` runs the runner on saved frames instead of the live screen.
- To pause, move the mouse to the top-left corner of the screen (`pause_corner`).
- To stop after the current game, create `data/stop_after_game.flag`.

### Configuration

`catanbot/config.py` holds the defaults. Anything you put in `data/config.json` overrides them.
The keys you're most likely to set:

| Key | Meaning |
| --- | --- |
| `bot_name` | Name the bot uses in chat |
| `friends` | Catan Universe names whose invites it accepts. An empty list accepts nobody. |
| `solo` | `true` starts a game against the computer from the home screen |
| `dry_run` | Decide moves but never click |
| `turn_budget_s` | How long its own turn may take |
| `imessage` | Local iMessage: group chat and owner handles (see below) |
| `facetime` | Local FaceTime voice: callers, speech and voice settings (see docs/facetime.md) |

### iMessage and FaceTime (optional)

Both are off until you configure them, and both run on the bot's own Mac with no server.

**iMessage.** Sign Messages in to the bot's Apple ID, then add this to `data/config.json`:

```json
"imessage": {"chat": "Catan night", "owners": ["+15550100"], "name": "marlow"}
```

- After each game the end screen goes to `chat` (a group chat's name or guid).
- An owner (a handle in `owners`) can write "marlow leave the game", "marlow score" or
  "marlow what's the plan" in that chat, or in a direct message. Anyone else is ignored, and the
  bot's hand is never sent.
- macOS asks for Automation permission (Messages) on the first send. Reading commands needs
  **Full Disk Access** for the app you start the bot from, because Messages keeps its messages in
  `~/Library/Messages/chat.db` (opened read-only).

**FaceTime.** Talk to the bot about the game on a call. Setup is in [docs/facetime.md](docs/facetime.md).

### Optional read-only API

`catanbot/api.py` serves the bot's state, current game and logs over HTTP and needs a bearer token.
Run it with Docker (above) or natively. `scripts/install_api.sh` installs it as a LaunchAgent for the
current user, so read the script before you run it.

The API reads these environment variables:

| Variable | Default |
| --- | --- |
| `CATANBOT_API_KEY_FILE` | `~/.catanbot-api.key` |
| `CATANBOT_API_PORT` | `8940` |

End-of-game sharing and remote commands are off unless you set `CATANBOT_SHARE_URL` or
`CATANBOT_COMMANDS_URL` and provide a key file.

## Layout

| Path | What's there |
| --- | --- |
| `catanbot/runner.py` | Live loop: read the screen, choose, click, verify |
| `catanbot/vision/` | Board, banners, rings and screen readers |
| `catanbot/mac.py` | Screen capture, input and OCR (Quartz + Vision) |
| `catanbot/brain.py`, `situational.py`, `rules.py` | Move choice and per-turn rule selection |
| `catanbot/sim.py`, `sim_cards.py` | Game simulator used for self-play and training |
| `catanbot/rag.py` | Strategy notes retrieval (`docs/strategy_*.md`) |
| `scripts/` | Self-play, battles, A/B tests, training |
| `docs/` | Rules research, strategy playbooks and analyses |
| `tests/` | pytest suite with recorded frames |
| `data/` | Model weights, calibration and RAG index. Runtime output is git-ignored. |

## License

MIT, see [LICENSE](LICENSE). Catan is a trademark of CATAN GmbH; this project is not affiliated with it.
