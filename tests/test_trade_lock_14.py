"""owner 2026-10-08: "when players have 14 points or more, trade should be rejected unless it gives
you a 100% win next round" -- and the wildcard panel that sat over our turn (live 18:34)."""
from pathlib import Path

import pytest
import cv2

from catanbot import moves as mv
from catanbot.board import hex_corners
from catanbot.runner import Runner

from test_canal_playbook import _gs, brain

FIX = Path(__file__).parent / "fixtures"
V = hex_corners(0, 0)[0]


def _leader_at(vp, me_vp=5, hand=None):
    gs = _gs()
    gs.players[1].vp_visible = vp
    me = gs.me()
    me.vp_visible = me_vp
    me.settlements.add(V)
    me.hand = dict(hand or {"wool": 2, "lumber": 1, "ore": 1})
    me.hand_count = sum(me.hand.values())
    return gs


def test_14_vp_partner_gets_no_ordinary_trade():
    br = brain()
    gs = _leader_at(14)
    assert br.trade_locked(gs, "leader", {"wool": 1}, {"ore": 1})
    assert not br.evaluate_offer(gs, "leader", {"wool": 1}, {"ore": 2})
    assert not br.trade_locked(_leader_at(13), "leader", {"wool": 1}, {"ore": 1})
    assert not br.trade_locked(gs, "weak", {"wool": 1}, {"ore": 1})


def test_14_vp_wildcard_is_declined():
    br = brain()
    gs = _leader_at(15)
    counters = [mv.Move("counter", give=(("wool", 1),), get=((r, 1),)) for r in ("lumber", "ore", "brick")]
    assert br.choose_answer(gs, counters + [mv.Move("decline")], "leader").kind == "decline"


def test_trade_that_wins_the_game_is_taken():
    br = brain()
    # 18 of 19 VP; the ore makes a city (2 grain + 3 ore) on our settlement
    gs = _leader_at(16, me_vp=18, hand={"grain": 2, "ore": 2, "wool": 1})
    assert br.sure_win_after(gs, {"wool": 1}, {"ore": 1})
    assert not br.trade_locked(gs, "leader", {"wool": 1}, {"ore": 1})
    assert br.evaluate_offer(gs, "leader", {"wool": 1}, {"ore": 1})
    assert not br.sure_win_after(gs, {"wool": 1}, {"brick": 1})


def test_our_offers_skip_a_14_vp_player():
    from catanbot.state import PlayerState
    br = brain()
    gs = _leader_at(16, hand={"ore": 3, "wool": 2, "lumber": 1})     # a city is 1 grain short
    gs.players.append(PlayerState("second", vp_visible=14))         # the leader is skipped anyway
    gs.me().cities.add(hex_corners(2, 0)[0])
    out = br.propose_trade(gs)
    assert out is not None and out[2], "no offer planned at all"
    assert "second" not in out[2] and "leader" not in out[2]
    br.trade_locked = lambda *a, **k: False         # the lock, not some other rule, kept them out
    assert "second" in br.propose_trade(gs)[2]


def test_wildcard_panel_red_x_found():
    img = cv2.imread(str(FIX / "wildcard_panel_183517.png"))
    if img is None:
        pytest.skip("live frame not in this checkout")
    hdr = {"text": "Complete the wildcard trade, change it, or decline it.", "x": 504, "y": 447, "w": 301, "h": 18}
    b = Runner.wildcard_decline_button(img, hdr)
    assert b is not None and abs(b[0] - 885) < 10 and abs(b[1] - 759) < 10
