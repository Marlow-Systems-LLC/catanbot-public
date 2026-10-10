// CatanbotFaceTime.app/Contents/MacOS/catanbot-facetime: the FaceTime daemon's launcher.
//
// Why a bundle: macOS (TCC) silently denies the microphone to a bare python started by launchd
// or sshd ("Policy disallows prompt"): there is no app to ask on behalf of. This tiny ad-hoc
// signed executable lives in a bundle with CFBundleIdentifier local.catanbot.facetime and an
// NSMicrophoneUsageDescription, and spawns the daemon's python as a CHILD (not exec): a child's
// microphone request is attributed to the responsible process, this bundle, which the owner
// allows once (docs/facetime.md "Microphone").
//
// It runs exactly CATANBOT_PY CATANBOT_SCRIPT [args...] (both set at build time by install.sh:
// the repo's .venv python and scripts/facetime_voice.py), forwards SIGTERM/SIGINT/SIGHUP, and
// exits with the child's status. Nothing else.
#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef CATANBOT_PY
#error "build with -DCATANBOT_PY=\"/path/to/.venv/bin/python\" (install.sh does)"
#endif
#ifndef CATANBOT_SCRIPT
#error "build with -DCATANBOT_SCRIPT=\"/path/to/scripts/facetime_voice.py\" (install.sh does)"
#endif

extern char **environ;
static volatile pid_t child = 0;

static void forward(int sig) {
  if (child > 0) kill(child, sig);
}

int main(int argc, char **argv) {
  char *py = CATANBOT_PY;
  char *script = CATANBOT_SCRIPT;

  char **args = calloc((size_t)argc + 2, sizeof *args);
  if (!args) return 71;
  args[0] = py;
  args[1] = script;
  for (int i = 1; i < argc; i++) args[i + 1] = argv[i];
  args[argc + 1] = NULL;

  struct sigaction sa = {0};
  sa.sa_handler = forward;
  sigaction(SIGTERM, &sa, NULL);
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGHUP, &sa, NULL);

  pid_t pid;
  int rc = posix_spawn(&pid, py, NULL, NULL, args, environ);
  if (rc != 0) {
    fprintf(stderr, "catanbot-facetime launcher: spawn %s failed: %d\n", py, rc);
    return 71;
  }
  child = pid;
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return 71;
  }
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}
