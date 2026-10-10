// facetime-ax.js — FaceTime through the Accessibility API (JXA).
//
//   osascript -l JavaScript facetime-ax.js detect [NotificationCenter,Messages]  → one JSON line
//   osascript -l JavaScript facetime-ax.js click <label> [window title]
//   osascript -l JavaScript facetime-ax.js dump     → every element's role/name (tuning)
//
// Run under `ssh localhost`: sshd holds the Accessibility
// grant, launchd-run processes do not (same trick as the imsg bridge's sends).
//
// ANSWER-ONLY BY CONSTRUCTION. This script can press exactly the buttons in
// CLICKABLE (answer/join an incoming call, leave the current one). It never
// opens a facetime:// or tel: URL, never types a number, never presses a
// call/dial button. test_facetime.py fails the build if any of that appears.
//
// macOS 27 (seen on a real ring, 2026-10-03): the banner's buttons have no
// names; the join is an AXButton described "Join FaceTime" (or "Join") in
// the chat's Messages window. A label matches a button's name or description
// exactly, never a substring: "FaceTime" and "FaceTime Audio" place calls.
// `detect` reports raw texts, per-window clickable labels and (when a
// FaceTime text shows) per-banner Notification Center groups (never Messages
// texts: those are the chat), the daemon decides; `dump` is for tuning.

/* global ObjC, Application */
// `run` is the JXA entry point osascript calls; nothing in this file calls it.
ObjC.import("stdlib");

const CLICKABLE = ["Join FaceTime", "Join", "Accept", "Answer", "Leave", "End"];
const PROCS = ["NotificationCenter", "FaceTime", "Messages"];
const NO_TEXTS = ["Messages"];

function se() {
  return Application("System Events");
}

function procIfRunning(name) {
  const p = se().processes.whose({ name: name });
  return p.length ? p[0] : null;
}

function safe(f, d) {
  try {
    const v = f();
    return v === undefined || v === null ? d : v;
  } catch {
    return d;
  }
}

// Every UI element of a process's windows, flattened (entireContents is one
// Apple Event per window: fast enough for a 1.5 s poll).
function elements(p) {
  const out = [];
  const wins = safe(() => p.windows(), []);
  for (const w of wins) {
    out.push({ el: w, win: w });
    for (const e of safe(() => w.entireContents(), [])) out.push({ el: e, win: w });
  }
  return out;
}

// withValue false: never read an element's value (Messages: that is chat text).
function info(e, withValue) {
  const name = String(safe(() => e.name(), "") || "");
  const desc = String(safe(() => e.description(), "") || "");
  return {
    role: safe(() => e.role(), ""),
    name: name || desc,
    labels: [name, desc].filter((x) => x),
    value: withValue ? String(safe(() => e.value(), "") || "") : "",
  };
}

function clickableLabels(i) {
  return i.labels.filter((l) => CLICKABLE.indexOf(l) >= 0);
}

// Notification Center banners, each with only its OWN texts and clickable
// labels, so a 1:1 caller is read from the ringing banner and never from a
// neighbouring notification. A banner is a node with static-text children;
// its buttons are those in its subtree, minus nested banners. One Apple Event
// per node, so detect only walks this when a FaceTime text is on screen.
function ncGroups(p) {
  const groups = [];
  let budget = 400;
  function walk(node, depth) {
    if (depth > 14 || budget-- <= 0) return null;
    const kids = safe(() => node.uiElements(), []);
    const texts = [];
    const buttons = [];
    for (const k of kids) {
      const i = info(k, true);
      if (i.role === "AXStaticText" && (i.value || i.name)) texts.push((i.value || i.name).slice(0, 120));
      else if (i.role === "AXButton") {
        for (const l of clickableLabels(i)) if (buttons.indexOf(l) < 0) buttons.push(l);
      }
    }
    const own = { texts: texts.slice(0, 12), buttons: buttons };
    for (const k of kids) {
      const r = walk(k, depth + 1);
      if (!r) continue;
      if (!r.banner) for (const l of r.buttons) if (own.buttons.indexOf(l) < 0) own.buttons.push(l);
    }
    if (texts.length) {
      groups.push(own);
      return { banner: true, buttons: own.buttons };
    }
    return { banner: false, buttons: own.buttons };
  }
  for (const w of safe(() => p.windows(), [])) walk(w, 0);
  return groups.slice(0, 20);
}

// only: comma-separated subset of PROCS. The daemon leaves FaceTime out while
// no call is up: after a missed call FaceTime opens, and walking its window
// takes >10 s on macOS 27, so every detect would time out until it quits.
function detect(only) {
  const res = { procs: {}, ts: Date.now() };
  const want = only ? only.split(",").filter((p) => PROCS.indexOf(p) >= 0) : PROCS;
  for (const pn of want) {
    const p = procIfRunning(pn);
    if (!p) continue;
    const texts = [];
    const buttons = [];
    const wins = [];
    let cur = null;
    for (const { el, win } of elements(p)) {
      const i = info(el, NO_TEXTS.indexOf(pn) < 0);
      if (el === win) {
        cur = { title: String(safe(() => win.title(), "") || i.name).slice(0, 200), buttons: [] };
        wins.push(cur);
      } else if (i.role === "AXButton") {
        if (i.name && NO_TEXTS.indexOf(pn) < 0) buttons.push(i.name);
        for (const l of clickableLabels(i)) if (cur && cur.buttons.indexOf(l) < 0) cur.buttons.push(l);
      } else if (i.role === "AXStaticText" && (i.value || i.name) && NO_TEXTS.indexOf(pn) < 0) texts.push((i.value || i.name).slice(0, 120));
    }
    res.procs[pn] = { windows: wins.length, wins: wins.slice(0, 10), texts: texts.slice(0, 40), buttons: buttons.slice(0, 40) };
    if (pn === "NotificationCenter" && texts.some((t) => /facetime/i.test(t))) res.procs[pn].groups = ncGroups(p);
  }
  return res;
}

// window: only press inside the window with exactly this title (the chat the
// daemon matched); omitted, the first button with this label anywhere.
function click(label, window) {
  if (CLICKABLE.indexOf(label) < 0) return { ok: false, error: "not an answer/leave button: " + label };
  for (const pn of PROCS) {
    const p = procIfRunning(pn);
    if (!p) continue;
    for (const { el, win } of elements(p)) {
      if (window && String(safe(() => win.title(), "") || "") !== window) continue;
      const i = info(el, false);
      if (i.role === "AXButton" && i.labels.indexOf(label) >= 0) {
        el.actions.byName("AXPress").perform();
        return { ok: true, proc: pn, label: label };
      }
    }
  }
  return { ok: false, error: "no button " + label };
}

function dump() {
  const out = [];
  for (const pn of PROCS) {
    const p = procIfRunning(pn);
    if (!p) continue;
    for (const { el } of elements(p)) {
      const i = info(el, NO_TEXTS.indexOf(pn) < 0);
      if (NO_TEXTS.indexOf(pn) >= 0 && i.role !== "AXButton" && i.role !== "AXWindow") continue;
      out.push(pn + "\t" + i.role + "\t" + i.labels.join(" | ") + "\t" + (NO_TEXTS.indexOf(pn) >= 0 ? "" : i.value.slice(0, 80)));
    }
  }
  return out.join("\n");
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function run(argv) {
  const cmd = argv[0] || "detect";
  if (cmd === "detect") return JSON.stringify(detect(argv[1] || ""));
  if (cmd === "click") return JSON.stringify(click(argv[1] || "", argv[2] || ""));
  if (cmd === "dump") return dump();
  return JSON.stringify({ ok: false, error: "usage: detect [procs] | click <label> [window] | dump" });
}
