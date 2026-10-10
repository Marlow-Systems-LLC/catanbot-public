"""macOS layer: find the Catan Universe window, capture it, click inside it, read text.

Needs Screen Recording (capture) and Accessibility (clicks) permission for the process
that runs the bot (Terminal or the venv python). Clicks are refused outside the game window.
"""
from __future__ import annotations

import time

import numpy as np
import Quartz
from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps  # noqa: F401

APP_OWNER = "Catan Universe"
BUNDLE = "unity.UnitedSoftMediaVerlagGmbH.CatanUniverse"


class WindowNotFound(RuntimeError):
    pass


def find_window() -> dict:
    wins = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionAll | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    best = None
    for w in wins:
        if w.get("kCGWindowOwnerName") != APP_OWNER or w.get("kCGWindowLayer", 1) != 0:
            continue
        b = w["kCGWindowBounds"]
        if b["Width"] < 400:
            continue
        if best is None or b["Width"] * b["Height"] > best["w"] * best["h"]:
            best = {"id": w["kCGWindowNumber"], "x": b["X"], "y": b["Y"], "w": b["Width"], "h": b["Height"],
                    "onscreen": bool(w.get("kCGWindowIsOnscreen")), "pid": w["kCGWindowOwnerPID"]}
    if not best:
        raise WindowNotFound("Catan Universe window not found — is the game running?")
    return best


TITLEBAR_PT = 28  # macOS title bar height in points (window mode)


def _bgr(img) -> np.ndarray:
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    bpr = Quartz.CGImageGetBytesPerRow(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    return np.frombuffer(data, dtype=np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3]  # BGRA -> BGR


def capture(win: dict | None = None) -> tuple[np.ndarray, dict]:
    """BGR image of the game window's content (title bar cropped) and the window info.
    Image pixels are Retina pixels; win['scale'] maps image px -> screen points."""
    win = win or find_window()
    img = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, win["id"],
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageNominalResolution * 0)
    if img is None and not win.get("onscreen"):
        # 2026-10-04: the game window sat off screen (other Space) and the capture came back None,
        # which read as a permission error; bring the game forward and try once more
        activate()
        time.sleep(1.0)
        win = find_window()
        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, win["id"],
            Quartz.kCGWindowImageBoundsIgnoreFraming)
    if img is None:
        if not Quartz.CGPreflightScreenCaptureAccess():
            raise WindowNotFound("capture failed — grant Screen Recording permission")
        raise WindowNotFound("capture failed: the game window is not on screen")
    arr = _bgr(img)
    if arr.max() == 0:
        # 21:30 2026-10-03: after a relaunch the window capture came back all black (every game
        # window "not on screen") while the screen showed the game; grab the screen region instead.
        # Only with the game in front: at 21:47 the desktop was in front and a Terminal's
        # "Last login" read as the game's login screen.
        if not frontmost_is_game():
            activate()
            time.sleep(1.0)
        if not frontmost_is_game() or not find_window().get("onscreen"):
            # 22:05: the game was "in front" but its window off screen; the region was the desktop
            raise WindowNotFound("game window capture is black and the window is not on screen")
        rect = Quartz.CGRectMake(win["x"], win["y"], win["w"], win["h"])
        img2 = Quartz.CGWindowListCreateImage(rect, Quartz.kCGWindowListOptionOnScreenOnly,
                                              Quartz.kCGNullWindowID, 0)
        if img2 is not None:
            arr = _bgr(img2)
    w = arr.shape[1]
    scale = w / win["w"]
    # crop the title bar if the window has one (window mode)
    tb = int(round(TITLEBAR_PT * scale)) if win["h"] < 2000 and _has_titlebar(arr, scale) else 0
    win = dict(win, scale=scale, crop_top_px=tb)
    return np.ascontiguousarray(arr[tb:]), win


def _has_titlebar(arr: np.ndarray, scale: float) -> bool:
    band = arr[: int(20 * scale)]
    return float(band.std()) < 25 and float(band.mean()) > 30


def activate() -> None:
    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(BUNDLE)
    if apps:
        apps[0].activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        time.sleep(0.25)
    if not frontmost_is_game():
        # newer macOS often refuses activation from a background process; LaunchServices doesn't
        import subprocess
        subprocess.run(["open", "-b", BUNDLE], check=False)
        for _ in range(10):
            time.sleep(0.15)
            if frontmost_is_game():
                break
    time.sleep(0.3)   # Unity drops the first click after it gains focus


def frontmost_is_game() -> bool:
    # NSWorkspace caches the front app in a process without a run loop (stale when the
    # owner types in Claude); ask LaunchServices directly instead.
    import subprocess
    try:
        front = subprocess.run(["lsappinfo", "front"], capture_output=True, text=True, timeout=2).stdout.strip()
        info = subprocess.run(["lsappinfo", "info", "-only", "bundleid", front],
                              capture_output=True, text=True, timeout=2).stdout
        return BUNDLE in info
    except Exception:
        return False


def screen_px() -> tuple[int, int]:
    """Main display size in pixels (full screen check, 2026-10-03)."""
    d = Quartz.CGMainDisplayID()
    return int(Quartz.CGDisplayPixelsWide(d)), int(Quartz.CGDisplayPixelsHigh(d))


def to_screen(win: dict, px: float, py: float) -> tuple[float, float]:
    """Image pixel (of the cropped capture) -> global screen point."""
    s = win["scale"]
    return win["x"] + px / s, win["y"] + (py + win["crop_top_px"]) / s


def _post(evtype, x, y, button=Quartz.kCGMouseButtonLeft):
    ev = Quartz.CGEventCreateMouseEvent(None, evtype, (x, y), button)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def click(win: dict, px: float, py: float, hold: float = 0.08) -> None:
    """Click at an image-pixel location inside the game window. Brings the game to front first."""
    ih = win["h"] * win["scale"] - win["crop_top_px"]
    iw = win["w"] * win["scale"]
    if not (0 <= px < iw and 0 <= py < ih):
        raise ValueError(f"refusing click outside game window: {(px, py)}")
    for _ in range(3):              # the owner may be typing in another app: take focus back
        if frontmost_is_game():
            break
        activate()
    x, y = to_screen(win, px, py)
    _post(Quartz.kCGEventMouseMoved, x, y)
    time.sleep(0.04)
    _post(Quartz.kCGEventLeftMouseDown, x, y)
    time.sleep(hold)
    _post(Quartz.kCGEventLeftMouseUp, x, y)


def move(win: dict, px: float, py: float) -> None:
    """Move the pointer (no click) to an image-pixel location inside the game window."""
    ih = win["h"] * win["scale"] - win["crop_top_px"]
    iw = win["w"] * win["scale"]
    if not (0 <= px < iw and 0 <= py < ih):
        raise ValueError(f"refusing pointer move outside game window: {(px, py)}")
    _post(Quartz.kCGEventMouseMoved, *to_screen(win, px, py))


def drag(win: dict, p0, p1, steps: int = 12) -> None:
    if not frontmost_is_game():
        activate()
    x0, y0 = to_screen(win, *p0)
    x1, y1 = to_screen(win, *p1)
    _post(Quartz.kCGEventLeftMouseDown, x0, y0)
    for i in range(1, steps + 1):
        t = i / steps
        _post(Quartz.kCGEventLeftMouseDragged, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        time.sleep(0.015)
    _post(Quartz.kCGEventLeftMouseUp, x1, y1)


def key(keycode: int) -> None:
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.02)


def type_text(text: str) -> None:
    for ch in text:
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(None, 0, down)
            Quartz.CGEventKeyboardSetUnicodeString(ev, len(ch), ch)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.01)


# ---- OCR (Apple Vision, on-device) --------------------------------------------------
def ocr(img: np.ndarray, fast: bool = False, min_conf: float = 0.3) -> list[dict]:
    """Return [{text, x, y, w, h, conf}] in image pixels (top-left origin).
    stage 2026-10-10 05:19: Vision went silent after ~12 h in one process (UNCONFIRMED cause):
    each call drains its own autorelease pool, and a failed request is printed, not swallowed."""
    import objc
    with objc.autorelease_pool():
        return _ocr(img, fast, min_conf)


def _ocr(img, fast, min_conf):
    import Vision
    from Foundation import NSData
    import cv2
    ok, png = cv2.imencode(".png", img)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(1 if fast else 0)
    req.setUsesLanguageCorrection_(False)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        print(f"OCR: Vision request failed: {err}", flush=True)
    H, W = img.shape[:2]
    out = []
    for r in req.results() or []:
        cand = r.topCandidates_(1)
        if not cand:
            continue
        c = cand[0]
        if c.confidence() < min_conf:
            continue
        bb = r.boundingBox()  # normalised, bottom-left origin
        x, y = bb.origin.x * W, (1 - bb.origin.y - bb.size.height) * H
        out.append({"text": str(c.string()), "x": x, "y": y, "w": bb.size.width * W,
                    "h": bb.size.height * H, "conf": float(c.confidence())})
    return out


def permissions_ok() -> dict:
    res = {"screen_recording": bool(Quartz.CGPreflightScreenCaptureAccess())}
    try:
        from ApplicationServices import AXIsProcessTrusted
        res["accessibility"] = bool(AXIsProcessTrusted())
    except Exception:
        res["accessibility"] = None
    return res


def request_permissions() -> None:
    Quartz.CGRequestScreenCaptureAccess()
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    except Exception:
        pass
