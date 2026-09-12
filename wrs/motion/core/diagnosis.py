"""Why a planning call returned None -- an explicit, opt-in out-channel.

Every planning primitive keeps the library-wide contract "a MotionData, or
``None``": the return stays the single source of control flow, no exceptions
for ordinary infeasibility, no silent continuation. But a caller that wants to
REPAIR a failure -- retry with a larger budget, pick another grasp, insert a
regrasp, move the base -- needs to know WHY the None happened: an unreachable
pose calls for a different pose, a blocked one for a different world.

A :class:`Diagnosis` is that channel, passed EXPLICITLY (``diag=``) like the
collider -- a value the call site shows, never hidden state on the robot or a
global. Callers that don't care pass nothing and pay nothing. On failure the
primitive stamps ``stage`` (WHERE in its pipeline it gave up, from the small
vocabulary below) plus a human-readable ``detail`` and machine-readable
``counts``; on success the object stays falsy.

Stage vocabulary -- one entry per DISTINCT repair strategy, deliberately few:

    'ik'             no reachable, collision-free IK for the target pose.
                     ``counts`` splits candidates / invalid / rejected, so
                     zero candidates reads "unreachable" (change the pose or
                     the base) while all-invalid reads "blocked" (change the
                     world or the grasp).
    'start_invalid'  the start config itself violates bounds / constraints /
                     collision (the world is mis-set-up, not the goal).
    'goal_invalid'   an explicit config goal does (same repair as above).
    'rrt'            the probabilistic search found no path in its budget
                     (retry with a larger budget or another goal branch).
    'joint_line'     the straight joint-space travel leg is blocked.
    'cartesian'      the straight cartesian leg (descent / retreat / insert)
                     broke -- an IK discontinuity or a collision along the
                     line; ``detail`` names the leg and the waypoint.

When a rejection came from a :class:`~wrs.motion.core.constraint.Constraint`
rather than a collision, the constraint's ``last_fail`` reason string (its own
diagnostic contract) is folded into ``detail`` via :func:`constraint_detail`
-- "blocked by the wall" and "blocked by the cable length" call for different
repairs.

:class:`~wrs.manipulation.recipe.Recipe` threads one Diagnosis per motion step
automatically and exposes the failed step's record as ``recipe.failure``.
"""


def constraint_detail(constraints):
    """Why a constraint rejected the last validity check: the FIRST non-None
    ``last_fail`` in evaluation order, or ''. Read RIGHT AFTER the failed
    check: evaluation short-circuits at the rejector (everything before it
    passed and cleared its ``last_fail``), so the first reason found is the
    actual rejector -- later entries may be stale from earlier calls. An
    all-None scan means the rejection was collision/bounds, not a constraint."""
    for c in constraints:
        reason = getattr(c, 'last_fail', None)
        if reason:
            return reason
    return ''


class Diagnosis:
    """Where and why one planning call failed. Falsy until ``fail`` stamps it."""

    def __init__(self, step=None):
        self.step = step      # recipe-level step label, e.g. '1:moveto'
        self.stage = None     # where the pipeline gave up (module vocabulary)
        self.detail = ''      # human-readable specifics
        self.counts = {}      # machine-readable numbers (e.g. ik candidates)

    def fail(self, stage, detail='', **counts):
        """Stamp the failure. The primitive STILL returns None -- the return
        stays the only control flow; this is purely the explanation."""
        self.stage = stage
        self.detail = detail
        self.counts = dict(counts)

    def __bool__(self):
        return self.stage is not None

    def __repr__(self):
        if not self:
            return '<Diagnosis ok>'
        parts = ([f'step={self.step!r}'] if self.step is not None else [])
        parts.append(f'stage={self.stage!r}')
        if self.detail:
            parts.append(f'detail={self.detail!r}')
        parts += [f'{k}={v}' for k, v in self.counts.items()]
        return f"<Diagnosis {' '.join(parts)}>"
