# FaceTime voice add-on (optional, macOS, off by default)

The bot can answer FaceTime audio calls from people you list and talk about the game it is
playing: the score, where the game is, its plan, why it made a move. It can also leave the game
when you ask it to. Everything runs on this Mac. No server is involved, and the bot never places a call.

It is off unless `data/config.json` has a `"facetime"` block. With `"answer": false` (the default)
it only logs "would answer" for allowed calls and lets them ring.

```
FaceTime ring (group: Messages "Join FaceTime"; 1:1: the "<name>, FaceTime Audio" banner)
  -> local allow-list (callers / groups in config)       anything else rings out
  -> press Join FaceTime / Join / Answer / Accept         only with "answer": true
in the call:
  BlackHole 16ch (FaceTime's speaker) -> voice detection -> speech-to-text (your STT server)
  -> wake rule ("marlow, ..." or the configured name) -> brain (your LLM, or built-in rules)
  -> text-to-speech (macOS `say`, or your TTS server) -> BlackHole 2ch (FaceTime's microphone)
```

Code: `catanbot/facetime/` (lib, config, brain, daemon, `macos/` helpers), entry point
`scripts/facetime_voice.py`, tests `tests/test_facetime*.py`.

## What it says

- **Score / where are we**: the public status line from `data/game_logs/current.json`.
- **Plan**: the bot's plan, goals and threats.
- **Why**: the last logged decision.
- **Leave the game** (for example "marlow, leave the game" or "quit the match"): writes
  `data/leave.flag`, exactly like the iMessage leave command, and the runner walks out of the game.
- **Hang up** (for example "marlow, hang up", or "bye" on a 1:1 call): ends the call. The game is not touched.

With an `llm` configured, the same game context goes to your model and it answers in its own
words. Without one, or if the model fails, the built-in rules answer. The hand is never spoken.
Every reply is scrubbed of card counts before speech, and the model is never sent the hand.

## Install

1. **BlackHole** (two virtual audio devices):

   ```
   brew install blackhole-2ch blackhole-16ch
   ```

   macOS may need its audio service restarted before the new devices appear. Run this yourself
   in Terminal (it needs your password): `sudo killall coreaudiod`.

2. **FaceTime audio settings.** In FaceTime, during a call or under Video > Microphone / Output:
   - Microphone: **BlackHole 2ch** (the bot's voice goes in here)
   - Output: **BlackHole 16ch** (the bot hears callers here)

   While the add-on runs, you will not hear FaceTime on your speakers. This Mac becomes the bot's line.

3. **Remote Login and Accessibility.** The ring and button helpers run under `ssh localhost`,
   because sshd holds the Accessibility grant and survives restarts.
   - System Settings > General > Sharing > Remote Login: on, for your user only.
   - Your own SSH key must be in `~/.ssh/authorized_keys`. Check that
     `ssh -o BatchMode=yes localhost true` works.
   - System Settings > Privacy & Security > Accessibility: grant sshd. On recent macOS the entry
     is `/usr/libexec/sshd-keygen-wrapper`, but this is unconfirmed: check which entry appears after
     the first `ssh localhost osascript ...` run is refused.

   Alternatively, set `"tuning": {"ax_via": "direct"}` and grant Accessibility to the app that runs
   the daemon instead.

4. **Config.** Add the block below to `data/config.json`, then run
   `.venv/bin/python scripts/facetime_voice.py --check`. It prints a summary of the config and never the keys.

5. **Install** (requires the repo `.venv` and Xcode command line tools):

   ```
   catanbot/facetime/macos/install.sh            # dry run: lists what is missing
   catanbot/facetime/macos/install.sh --apply
   ```

   It installs `requirements-facetime.txt` into `.venv` and runs the tests. It then builds the
   `facetime-nc` helper and `CatanbotFaceTime.app` into `~/.catanbot-facetime/`, and loads the
   LaunchAgent `local.catanbot.facetime`. Logs go to `~/.catanbot-facetime/facetime.log`.

6. **Microphone permission**, once, from a logged-in desktop session:

   ```
   open -n -W ~/.catanbot-facetime/CatanbotFaceTime.app --args --mic-check
   tail -1 ~/.catanbot-facetime/mic-check.log
   ```

   Click **Allow** when macOS asks. A plain python started by launchd is silently denied the
   microphone, which is why the app bundle exists. If `launcher.c`, `Info.plist` or the repo path
   changes, the app is rebuilt and macOS asks again.

To remove the add-on: `launchctl bootout gui/$(id -u)/local.catanbot.facetime`, then delete
`~/Library/LaunchAgents/local.catanbot.facetime.plist` and `~/.catanbot-facetime/`.

## Config

```json
{
  "facetime": {
    "answer": false,
    "name": "marlow",
    "callers": {"Ana Pop": "+15555550100", "Bo": "bo@example.com"},
    "groups": {"Catan": ["+15555550100", "bo@example.com", "+15555550177"]},
    "stt_url": "http://127.0.0.1:8080/v1/audio/transcriptions",
    "stt_model": "whisper-1",
    "llm": {"url": "http://127.0.0.1:11434/v1", "model": "llama3.1", "key_file": ""},
    "tts": "say",
    "voice": "Samantha",
    "in_device": "BlackHole 16ch",
    "out_device": "BlackHole 2ch"
  }
}
```

| Key | Meaning |
| --- | --- |
| `answer` | Only a literal `true` presses answer buttons. Anything else means log only. |
| `name` | The spoken wake name. "marlow" also accepts the common speech-to-text spellings (Marlo, Marlowe, Marlău, ...). |
| `callers` | Name as FaceTime shows it, mapped to a phone number or email. A 1:1 call is answered only from these. |
| `groups` | Optional named member lists. A group call is answered when every participant FaceTime shows resolves to a handle and either fits one group (the list may hold one more member: this Mac's own account) or all of them are in `callers`. |
| `stt_url`, `stt_model`, `stt_key_file` | An OpenAI-compatible `/audio/transcriptions` endpoint, for example a local whisper.cpp or faster-whisper server. Required for listening. |
| `llm` | Optional. `url` is an OpenAI-compatible base URL (`/chat/completions` is appended), plus `model` and `key_file` (a file holding a bearer key; empty for local servers). |
| `tts`, `tts_model`, `tts_key_file`, `voice` | `"say"` uses macOS speech with `voice`. A URL uses an OpenAI-compatible `/audio/speech` endpoint. Replies that look Romanian use the `say` voice in `tuning.ro_voice`. |
| `in_device`, `out_device` | Audio devices: FaceTime's output, and FaceTime's microphone. |
| `greeting` | `false` turns off the 1:1 greeting ("Hi, it's Marlow."). |
| `max_call_s` | Leave any call after this long (default 3600). |
| `tuning` | Optional knobs: `vad_floor` (lower it, to about 40, for quiet callers), `vad_tail_s`, `barge_floor`, `followup_s`, `brain_deadline_s`, `silent_end_s`, `ax_via`, `stt_prompt`, `dm_greeting`, `group_greeting` and more. See `catanbot/facetime/config.py`. |

Keys are read from files and never logged. Audio stays in memory for one utterance. Transcripts
and replies are never logged; the log holds only lengths, counts and timings.

## Tools

```
scripts/facetime_voice.py --check                     config summary (no secrets)
scripts/facetime_voice.py --ask "marlow who's winning" one brain reply as JSON (no audio, no FaceTime)
scripts/facetime_voice.py --in q.wav --out a.wav --brain stub|local   whole pipeline on a WAV file
scripts/facetime_voice.py --level 10                  input level meter (tune vad_floor)
```

## Safety

- Answer only. `facetime-ax.js` can press only Join FaceTime, Join, Accept, Answer, Leave and End.
  `facetime-nc` can press only Answer and Accept, and only inside the single FaceTime banner that
  names an allowed caller. There is no call-placing code. The tests check the sources for
  `facetime://`, `tel:`, URL opening and dial buttons.
- Unknown or ambiguous callers ring out. Nothing is guessed: two FaceTime banners, two names, or
  any unknown participant all mean no answer.
- If audio fails after answering, the bot hangs up instead of sitting in a call deaf.
- After "hang up" (or a silent call) it does not rejoin the same ring.

## UNCONFIRMED

These parts were ported from a setup that ran live, but have not been run live in this repository:

- **Live answering has not been tested here.** The daemon was exercised only with fake FaceTime,
  audio and speech objects (`tests/test_facetime_daemon.py`). No real call was placed or answered
  during the port.
- **Audio through BlackHole** (capture, playback, barge-in levels, the microphone grant for
  `CatanbotFaceTime.app`) has not been run from this repository.
- **macOS 27 accessibility labels.** Ring detection relies on FaceTime/NotificationCenter element
  names seen on macOS 27 (the 1:1 fixture is an anonymised real capture). Other macOS versions
  may label them differently.
- **Call end.** FaceTime can hide its in-call controls. The bot treats "no call controls" as
  unknown while there was audio in the last 20 s. This heuristic replaces a check the original
  setup did differently and is untested live. Leave/End may not be visible, in which case the bot
  quits FaceTime to end the call (`tuning.quit_to_leave`).
- **Group allow-list counts.** The "one more member than shown" rule assumes FaceTime always
  leaves out this Mac's own account.
- **STT, LLM and TTS endpoints.** Only the request format was tested (with fakes). No real
  whisper, LLM or speech server was called.
