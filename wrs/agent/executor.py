"""Execute one generated plan script against a prepared scene namespace.

The contract mirrors the library's planning contract: the script's deliverable
is the variable ``motion`` -- a MotionData on success, None on failure (the
script CHECKS every step return and stops, no silent continuation). This
module runs the script, then gathers everything a reviser needs to know:

  * ``error``    -- the traceback if the script itself crashed
  * ``failures`` -- every stamped :class:`~wrs.motion.core.diagnosis.Diagnosis`
                    found in the namespace (a Recipe left behind by a failed
                    step carries one on ``.failure``) -- harvested
                    mechanically, so the script does not have to remember to
                    report why it failed
  * ``stdout``   -- whatever the script printed

The script is executed IN-PROCESS with full access to the scene objects --
that is the point (the plan drives the real planning stack), and the reason
this executor is for SIMULATION plans from a trusted loop only, not a sandbox
for arbitrary code.
"""
import contextlib
import io
import traceback


class Outcome:
    """What one script run produced; falsy-``ok`` outcomes carry the why."""

    def __init__(self):
        self.motion = None    # the script's deliverable (MotionData or None)
        self.error = ''       # traceback text if the script raised
        self.failures = []    # stamped Diagnosis records found in the namespace
        self.stdout = ''
        self.namespace = {}   # the post-run namespace (for a success check)

    @property
    def ok(self):
        return self.error == '' and self.motion is not None

    def describe(self):
        """The failure, written for the reviser (one string, most useful
        signal first): the planning Diagnosis when there is one, else the
        traceback, else the bare fact that ``motion`` is unset."""
        if self.ok:
            return 'ok'
        parts = []
        if self.failures:
            parts.append('planning failed: ' +
                         '; '.join(repr(f) for f in self.failures))
        if self.error:
            parts.append('the script raised:\n' + self.error)
        if not parts:
            parts.append("the script completed but `motion` is None and no "
                         "step stamped a Diagnosis -- did it forget to plan, "
                         "or to assign `motion`?")
        if self.stdout.strip():
            parts.append('stdout:\n' + self.stdout.strip())
        return '\n'.join(parts)


def run_script(code, namespace):
    """Run ``code`` against a copy of ``namespace``; return the Outcome.

    The copy keeps the caller's dict clean (the scene OBJECTS are still
    shared and mutate -- rebuild the scene per attempt for a fresh start).
    """
    out = Outcome()
    buf = io.StringIO()
    ns = dict(namespace)
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, '<plan>', 'exec'), ns)
    except Exception:
        out.error = traceback.format_exc(limit=8)
    out.stdout = buf.getvalue()
    out.namespace = ns
    out.motion = ns.get('motion')
    for v in ns.values():
        failure = getattr(v, 'failure', None)
        if failure:                        # a Diagnosis is falsy until stamped
            out.failures.append(failure)
    return out
