"""Owner 2026-10-04: live feed to marlow.systems, public state only, off by default."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from catanbot import live_feed
from catanbot.state import Knight
from tests.test_road_cuts import _gs

SECRET = ("grain", "ore", "medicine", "hand", "plan", "progress_hand", "estimate", "target")


def _rich_gs():
    gs, me, o = _gs()
    me.hand = {"grain": 3, "ore": 2}
    me.progress_hand = ["medicine"]
    o.hand = {"ore": 1}                                   # an estimate of a rival's hand
    me.knights[gs.board.vertices[0]] = Knight(level=2, active=True)
    return gs


def test_snapshot_is_public_only():
    snap = live_feed.snapshot(_rich_gs(), "20261004_040000", "live", [8, 7], ["a 7 was rolled"])
    assert set(snap) <= live_feed.TOP_KEYS
    assert all(set(p) == live_feed.PLAYER_KEYS for p in snap["players"])
    body = json.dumps({k: v for k, v in snap.items() if k != "events"}).lower()
    assert not any(s in body for s in SECRET)
    me = snap["players"][0]
    assert me["knights"] == {"total": 1, "active": 1, "strength": 2}
    assert snap["turn"] == 2 and snap["recentDice"] == [8, 7] and "result" not in snap


def test_finished_carries_result():
    snap = live_feed.snapshot(_rich_gs(), "g", "finished", [], [], {"winner": "rival", "ranking": []})
    assert snap["result"]["winner"] == "rival"


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(live_feed, "CONF", tmp_path / "feed.json")
    monkeypatch.delenv("CATAN_FEED_URL", raising=False)
    monkeypatch.delenv("CATAN_FEED_KEY", raising=False)
    assert live_feed.config() is None and live_feed.post({"v": 1}) is None
    (tmp_path / "feed.json").write_text(json.dumps({"enabled": False, "url": "http://x", "key": "k"}))
    assert live_feed.config() is None


def test_post_sends_bearer_json():
    got = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got["auth"] = self.headers["Authorization"]
            got["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(204); self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    notes = []
    t = live_feed.post({"v": 1}, {"url": f"http://127.0.0.1:{srv.server_port}/", "key": "test-key"}, notes.append)
    t.join(5)
    srv.server_close()
    assert got == {"auth": "Bearer test-key", "body": {"v": 1}} and notes == []


def test_runner_feed_never_raises_and_skips_when_off(monkeypatch):
    from catanbot import runner as R
    r = R.Runner.__new__(R.Runner)
    r.io, r.notes, r.gs = object(), [], _rich_gs()
    r.note = r.notes.append
    monkeypatch.setattr(live_feed, "config", lambda: None)
    r._feed("live")
    assert r.notes == [] and r._feed_conf is None
    r._feed_conf = {"url": "http://127.0.0.1:9/", "key": "k"}
    monkeypatch.setattr(live_feed, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    r._feed("live")                                       # a failure is only a note
    assert r.notes == ["feed: RuntimeError"]
