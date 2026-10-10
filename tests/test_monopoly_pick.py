"""Solo 2026-10-05 06:12: the Resource Monopoly picker has icon tiles only; the bot never named a
resource and End turn stayed grey for 13 min."""
from board_fixture import canvas
from catanbot import runner as R

# dialog crops of data/probe/062048_033_rescue_240_native.png, data/snaps/223759.png and
# data/probe/064715_023_progress_after_merchant_fleet.png, pasted back into their full frames
SNAP = "monopoly_062048_1920x1080_420_460.png"


class FakeIO:
    def __init__(self, frames):
        self.frames, self.clicks = list(frames), []

    def snapshot(self):
        return self.frames.pop(0) if self.frames else None

    def ocr(self, img):
        from catanbot import mac
        return mac.ocr(img)


def test_monopoly_pick_clicks_grain_tile_then_check():
    img = R.to_frame(canvas(SNAP))
    r = R.Runner.__new__(R.Runner)
    clicks = []
    r.click = lambda xy, why="", ocr=None: clicks.append((xy, why))
    r.note = lambda *a, **k: None
    r.io = FakeIO([img])
    r.probe = type("P", (), {"result": lambda *a, **k: None})()
    from catanbot import mac
    ocr = mac.ocr(img)
    assert r._monopoly_title(ocr) is not None
    r._monopoly_pick("grain", ocr)
    (gx, gy), _ = clicks[0]
    (kx, ky), _ = clicks[1]
    # grain tile and green check, frame px measured on the live frame (view 526,407 / 786,592)
    assert abs(gx - 656) < 15 and abs(gy - 584) < 15
    assert abs(kx - 980) < 15 and abs(ky - 850) < 15


LIVE = "monopoly_223759_1920x1080_420_460.png"


def test_monopoly_pick_live_virtual_frame():
    """Solo 22:37: the live 1308x790 virtual frame (UI x1.468) -- clicks were 33/50 px short."""
    from catanbot import mac
    from catanbot.vision.layout import Layout
    nat = canvas(LIVE)
    lay = Layout(nat.shape[1], nat.shape[0])
    img = lay.frame(nat, R.FRAME)
    r = R.Runner.__new__(R.Runner)
    clicks = []
    r.click = lambda xy, why="", ocr=None: clicks.append((xy, why))
    r.note = lambda *a, **k: None
    r.io = FakeIO([img])
    r.io.lay = lay
    r.probe = type("P", (), {"result": lambda *a, **k: None})()
    r._monopoly_pick("ore", mac.ocr(img))
    (ox, oy), (kx, ky) = (lay.to_native(*c[0]) for c in clicks[:2])
    # native px on the 22:37 screen: ore tile ~(1031, 700-725), green check ~(1370, 1035)
    assert abs(ox - 1031) < 25 and 680 < oy < 745
    assert abs(kx - 1370) < 25 and abs(ky - 1035) < 25


FLEET = "fleet_064715_1372x888_290_360.png"


def test_merchant_fleet_dialog_uses_the_tile_row():
    """Solo 2026-10-06 06:47: Merchant Fleet's picker is the Monopoly tile row under its own title;
    OCR found no 'lumber' and the dialog stalled 17 min. Lumber tile, then the green check."""
    img = R.to_frame(canvas(FLEET))
    r = R.Runner.__new__(R.Runner)
    clicks, started = [], []
    r.click = lambda xy, why="", ocr=None: clicks.append((xy, why))
    r.note = lambda *a, **k: None
    r.io = FakeIO([img])
    # stage's live virtual frame: UI x1.468 (as at 22:37), so 0.681 frame px per native px
    r.io.lay = type("L", (), {"h": 1080, "s": 1.468})()
    r.probe = type("P", (), {"result": lambda *a, **k: None})()
    r.brain = type("B", (), {"start_fleet": lambda self, x: started.append(x), "fleet_resource": None})()
    from catanbot import mac
    t = r._monopoly_title(mac.ocr(img))
    assert t is not None and "Fleet" in t["text"]
    r._monopoly_pick("lumber", None)
    (lx, ly), _ = clicks[0]
    (kx, ky), _ = clicks[1]
    # frame px measured on the probe: lumber tile (389, 538), green check (934, 758)
    assert abs(lx - 389) < 15 and abs(ly - 538) < 15
    assert abs(kx - 934) < 15 and abs(ky - 758) < 15
    assert started == []                     # dialog still up in the replay (no new frame)
