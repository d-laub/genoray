/* Runs a command with transparent huge pages DISABLED for the process.
 *
 * `PR_SET_THP_DISABLE` is preserved across `execve(2)` (prctl(2)), so this
 * wrapper puts an ALREADY-BUILT binary into the degraded-TLB regime without
 * touching its source. That matters here: the two #177 A/B binaries were
 * already built and measured, and adding an in-process knob would have
 * changed them and invalidated the comparison against the published numbers.
 *
 * Unprivileged -- no root, no writes to /sys/kernel/mm/transparent_hugepage.
 *
 *   cc -O2 -o nothp nothp.c
 *   ./nothp ./some_binary --args
 */
#include <stdio.h>
#include <sys/prctl.h>
#include <unistd.h>

#ifndef PR_SET_THP_DISABLE
#define PR_SET_THP_DISABLE 41
#endif
#ifndef PR_GET_THP_DISABLE
#define PR_GET_THP_DISABLE 42
#endif

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: nothp <command> [args...]\n");
    return 2;
  }
  if (prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0) {
    perror("prctl(PR_SET_THP_DISABLE)");
    return 3;
  }
  /* Read back rather than trusting the set: a silent no-op here would make
   * the THP-off arm secretly identical to the THP-on arm. */
  int got = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  if (got != 1) {
    fprintf(stderr, "nothp: THP_DISABLE reads back %d, expected 1\n", got);
    return 4;
  }
  fprintf(stderr, "nothp: THP disabled for this process (readback=1)\n");
  execvp(argv[1], argv + 1);
  perror("execvp");
  return 127;
}
