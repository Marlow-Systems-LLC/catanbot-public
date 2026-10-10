// facetime-nc.swift — the 1:1 FaceTime ring through the native Accessibility API.
//
//   facetime-nc detect                 → one JSON line
//   facetime-nc click <label> <caller> → one JSON line
//   facetime-nc incall [mic device]    → one JSON line (read-only: is a call up?)
//
// Built by install.sh (/usr/bin/swiftc), run under `ssh localhost` like
// facetime-ax.js: sshd holds the Accessibility grant.
//
// Why native: on macOS 27 the ring banner is a remote view inside
// NotificationCenter's "Notification Center" window. System Events (JXA) sees
// only empty nested groups there and takes 7 s+; AXUIElement follows the
// remote view and sees it in ~90 ms (captured 2026-10-04):
//   AXGroup desc="FaceTime Notification" id=FACETIME_NOTIFICATION
//     AXGroup
//       AXGenericElement desc="<LRE>Name<PDF>, FaceTime Audio"
//       AXButton desc="Live Voicemail" | "Decline with Message" | "Decline" | "Answer"
//
// ANSWER-ONLY BY CONSTRUCTION. `click` presses only a button described exactly
// "Answer" or "Accept", only inside the single FACETIME_NOTIFICATION group on
// screen, only when that group has exactly one caller label and it names the
// given caller, and only when exactly one such button is in it. It never
// presses anything else (Live Voicemail, Decline, a call button), never opens
// a URL and never types. test_facetime.py (NeverDials) checks this file.
// Every run is capped at HARD_CAP_S: it exits rather than hang.

import AppKit
import ApplicationServices
import CoreAudio
import Foundation

let HARD_CAP_S = 3.0
let PRESSABLE: Set<String> = ["Answer", "Accept"]
let GROUP_ID = "FACETIME_NOTIFICATION"
let MAX_NODES = 600
let MAX_DEPTH = 40

let t0 = Date()

let emitLock = NSLock()
var emitted = false

// Exactly one JSON line per run, even if the watchdog and the walk race.
func emit(_ obj: [String: Any]) {
  emitLock.lock()
  defer { emitLock.unlock() }
  if emitted { return }
  emitted = true
  var o = obj
  o["ms"] = Int(Date().timeIntervalSince(t0) * 1000)
  if let d = try? JSONSerialization.data(withJSONObject: o, options: [.sortedKeys]), let s = String(data: d, encoding: .utf8) {
    print(s)
  } else {
    print("{\"ok\":false,\"error\":\"json\"}")
  }
  fflush(stdout)
}

// The hard cap: a watchdog thread, plus SIGALRM (default action: terminate)
// in case the process is wedged beyond the reach of a thread.
alarm(UInt32(HARD_CAP_S) + 2)
DispatchQueue.global().asyncAfter(deadline: .now() + HARD_CAP_S) {
  emit(["ok": false, "error": "timeout"])
  exit(124)
}
// Each AX message gives up after 0.5 s (an unresponsive NotificationCenter).
AXUIElementSetMessagingTimeout(AXUIElementCreateSystemWide(), 0.5)

func attr(_ e: AXUIElement, _ a: String) -> AnyObject? {
  var v: AnyObject?
  return AXUIElementCopyAttributeValue(e, a as CFString, &v) == .success ? v : nil
}

func str(_ e: AXUIElement, _ a: String) -> String { (attr(e, a) as? String) ?? "" }

func kids(_ e: AXUIElement) -> [AXUIElement] { (attr(e, kAXChildrenAttribute) as? [AXUIElement]) ?? [] }

// Bidi/format marks around a contact name (U+200E/F, U+202A..U+202E,
// U+2066..U+2069, U+FEFF) go, any run of whitespace (macOS writes
// "FaceTime<U+00A0>Audio") becomes one space, then the ", FaceTime Audio"/
// ", FaceTime Video" suffix goes. Same rule as facetime_lib.caller_name.
func callerName(_ raw: String) -> String {
  var scalars = String.UnicodeScalarView()
  var space = false
  for s in raw.unicodeScalars {
    let v = s.value
    if v == 0x200E || v == 0x200F || (0x202A...0x202E).contains(v) || (0x2066...0x2069).contains(v) || v == 0xFEFF { continue }
    if CharacterSet.whitespacesAndNewlines.contains(s) {
      space = true
      continue
    }
    if space && !scalars.isEmpty { scalars.append(" ") }
    space = false
    scalars.append(s)
  }
  var t = String(scalars)
  for suffix in [", FaceTime Audio", ", FaceTime Video"] where t.hasSuffix(suffix) {
    t = String(t.dropLast(suffix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
  }
  return t
}

struct Banner {
  var callers: [String] = []  // raw AXGenericElement descriptions
  var buttons: [(String, AXUIElement)] = []  // AXButton description, element
}

var nodes = 0

func overBudget() -> Bool { nodes > MAX_NODES || Date().timeIntervalSince(t0) > HARD_CAP_S - 0.5 }

// One banner's own caller labels and buttons (a nested FaceTime group is its
// own banner, never merged into this one).
func collect(_ e: AXUIElement, _ depth: Int, _ b: inout Banner, _ out: inout [Banner]) {
  for k in kids(e) {
    if depth > MAX_DEPTH || overBudget() { return }
    nodes += 1
    if str(k, kAXIdentifierAttribute) == GROUP_ID {
      scanBanner(k, depth + 1, &out)
      continue
    }
    let role = str(k, kAXRoleAttribute)
    let desc = str(k, kAXDescriptionAttribute)
    if role == "AXGenericElement" && !desc.isEmpty { b.callers.append(desc) }
    if role == "AXButton" && !desc.isEmpty { b.buttons.append((desc, k)) }
    collect(k, depth + 1, &b, &out)
  }
}

func scanBanner(_ g: AXUIElement, _ depth: Int, _ out: inout [Banner]) {
  var b = Banner()
  var nested: [Banner] = []
  collect(g, depth, &b, &nested)
  out.append(b)
  out.append(contentsOf: nested)
}

func find(_ e: AXUIElement, _ depth: Int, _ out: inout [Banner]) {
  for k in kids(e) {
    if depth > MAX_DEPTH || overBudget() { return }
    nodes += 1
    if str(k, kAXIdentifierAttribute) == GROUP_ID {
      scanBanner(k, depth + 1, &out)
    } else {
      find(k, depth + 1, &out)
    }
  }
}

func banners() -> [Banner]? {
  guard AXIsProcessTrusted() else { return nil }
  guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.notificationcenterui").first else { return [] }
  let root = AXUIElementCreateApplication(app.processIdentifier)
  var out: [Banner] = []
  for w in (attr(root, kAXWindowsAttribute) as? [AXUIElement]) ?? [] where str(w, kAXTitleAttribute) != "Month" {
    if overBudget() { break }
    nodes += 1
    if str(w, kAXIdentifierAttribute) == GROUP_ID { scanBanner(w, 1, &out) } else { find(w, 1, &out) }
  }
  return out
}

func detect() {
  guard let bs = banners() else { emit(["ok": false, "error": "not trusted"]); return }
  emit([
    "ok": !overBudget(),
    "truncated": overBudget(),
    "groups": bs.map { ["callers": $0.callers.map { String($0.prefix(200)) }, "buttons": $0.buttons.map { String($0.0.prefix(80)) }] },
  ])
}

func click(_ label: String, _ caller: String) {
  guard PRESSABLE.contains(label) else { emit(["ok": false, "error": "not an answer button"]); return }
  let want = callerName(caller)
  guard !want.isEmpty else { emit(["ok": false, "error": "no caller"]); return }
  guard let bs = banners() else { emit(["ok": false, "error": "not trusted"]); return }
  guard !overBudget() else { emit(["ok": false, "error": "walk truncated"]); return }
  guard bs.count == 1 else { emit(["ok": false, "error": "\(bs.count) FaceTime banners"]); return }
  let b = bs[0]
  guard b.callers.count == 1 else { emit(["ok": false, "error": "\(b.callers.count) caller labels"]); return }
  guard callerName(b.callers[0]) == want else { emit(["ok": false, "error": "caller changed"]); return }
  let hits = b.buttons.filter { $0.0 == label }
  guard hits.count == 1 else { emit(["ok": false, "error": "\(hits.count) \(label) buttons"]); return }
  let r = AXUIElementPerformAction(hits[0].1, kAXPressAction as CFString)
  emit(r == .success ? ["ok": true, "label": label] : ["ok": false, "error": "AXPress \(r.rawValue)"])
}

// ── incall: is a call still up? READ-ONLY (never presses anything). ──────
// Two cheap signals, no JXA:
//  - AX: a button named/described exactly End or Leave (or "End Call"/"Leave
//    Call") in a window of FaceTime, InCallService or Messages, found by bundle
//    id (FaceTime may not run as an app: then that one is just absent).
//  - CoreAudio: FaceTime's microphone device (argv[2], "BlackHole 2ch") is
//    running somewhere. FaceTime holds it during a call; the bot opens it only
//    while speaking (lib.incall_state discounts that).
let CALL_APPS = ["com.apple.FaceTime", "com.apple.InCallService", "com.apple.MobileSMS"]
let END_LABELS: Set<String> = ["End", "Leave", "End Call", "Leave Call"]

func hasEnd(_ e: AXUIElement, _ depth: Int) -> Bool {
  for k in kids(e) {
    if depth > MAX_DEPTH || overBudget() { return false }
    nodes += 1
    if str(k, kAXRoleAttribute) == "AXButton" && (END_LABELS.contains(str(k, kAXDescriptionAttribute)) || END_LABELS.contains(str(k, kAXTitleAttribute))) { return true }
    if hasEnd(k, depth + 1) { return true }
  }
  return false
}

func micRunning(_ name: String) -> Bool? {
  let sys = AudioObjectID(kAudioObjectSystemObject)
  var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
  var size: UInt32 = 0
  guard AudioObjectGetPropertyDataSize(sys, &addr, 0, nil, &size) == noErr, size > 0 else { return nil }
  var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
  guard AudioObjectGetPropertyData(sys, &addr, 0, nil, &size, &ids) == noErr else { return nil }
  for id in ids {
    var nameAddr = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var cf: Unmanaged<CFString>?
    var sz = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(id, &nameAddr, 0, nil, &sz, &cf) == noErr, let n = cf?.takeRetainedValue() as String?, n == name else { continue }
    var runAddr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var running: UInt32 = 0
    var rs = UInt32(MemoryLayout<UInt32>.size)
    guard AudioObjectGetPropertyData(id, &runAddr, 0, nil, &rs, &running) == noErr else { return nil }
    return running != 0
  }
  return nil
}

func incall(_ micName: String?) {
  guard AXIsProcessTrusted() else { emit(["ok": false, "error": "not trusted"]); return }
  var found = false
  var running: [String] = []
  for bid in CALL_APPS {
    guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: bid).first else { continue }
    running.append(bid)
    let root = AXUIElementCreateApplication(app.processIdentifier)
    for w in (attr(root, kAXWindowsAttribute) as? [AXUIElement]) ?? [] {
      if overBudget() { break }
      nodes += 1
      if hasEnd(w, 1) { found = true; break }
    }
    if found { break }
  }
  var o: [String: Any] = ["ok": true, "incall": found, "truncated": !found && overBudget(), "apps": running]
  if let m = micName, !m.isEmpty { o["mic"] = micRunning(m).map { $0 as Any } ?? NSNull() }
  emit(o)
}

let argv = CommandLine.arguments
switch argv.count > 1 ? argv[1] : "detect" {
case "detect":
  detect()
case "click" where argv.count == 4:
  click(argv[2], argv[3])
case "incall":
  incall(argv.count > 2 ? argv[2] : nil)
default:
  emit(["ok": false, "error": "usage: detect | click <Answer|Accept> <caller> | incall [mic device]"])
}
exit(0)
