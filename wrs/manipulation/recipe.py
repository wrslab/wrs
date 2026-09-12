"""A manipulation Recipe: compose the primitive arm verbs into one motion.

Bind an arm + its collision world + constraints ONCE, then list steps -- the
free verbs ``moveto`` / ``linear`` (motion) and ``grasp`` / ``release`` (state).
Each step plans immediately against the current state; the Recipe threads the
start config between steps, tracks the gripper opening and the held object, and
accumulates the segments into one ``MotionData`` (``result``).

The motion steps RETURN their ``MotionData`` (or ``None`` if they -- or an
earlier step -- can't plan): check the return and stop, no fluent chaining and no
silent continuation. ``hold`` / ``release`` mutate the held state (no return).
WHY a step failed is on ``recipe.failure`` (a
:class:`~wrs.motion.core.diagnosis.Diagnosis` stamped by the primitive that gave
up) -- the repair channel for a caller that reacts to infeasibility (retry with a
larger budget, another grasp, an inserted regrasp) instead of just stopping.

This is the ergonomic layer over the orthogonal arm verbs (which stay directly
usable): every higher-level shape -- pick-and-place, pick-and-hold, regrasp's
depart-then-approach -- is just a different SEQUENCE of the same four steps, so
the library needs no combinatorial family of named compositions.

    r = arm.recipe(collider, constraints=[cable])    # or system.recipe(arm, aux=...)
    if r.moveto(pick_pose) is None:
        return None
    r.hold(obj, opening=jw)
    if r.moveto(goal_pose) is None:
        return None
    motion = r.result                                # MotionData (holds at goal)
"""
import numpy as np

from wrs.motion.core.diagnosis import Diagnosis
from wrs.motion.core.motion_data import MotionData


class Recipe:
    def __init__(self, arm, collider, *, constraints=(), tcp=None, start_qs=None):
        self.arm = arm
        self.collider = collider
        self.constraints = tuple(constraints)
        self._ee = arm.end_effector
        # default working frame: the EE grasp center (the frame poses target)
        self.tcp = tcp if tcp is not None else self._ee.tcp('grasp_center')
        self._start = (np.asarray(start_qs, dtype=np.float64).copy()
                       if start_qs is not None
                       else np.asarray(arm.body.qs, dtype=np.float64).copy())
        self._ee.open()                                # start open
        self._ee_qpos = np.asarray(self._ee.qs, dtype=np.float32).copy()
        # sync the open jaw into the collider so the reach legs collide against
        # the OPEN gripper (the gripper's finger joints are not the actor's qs, so
        # planning would otherwise see a stale opening from a previous recipe).
        self.collider.set_mecba_qpos(self._ee, self._ee_qpos)
        self._held = None
        self._segs = []
        self._failed = False
        self._failure = None

    # ---- motion steps (return the planned MotionData, or None) ---------------
    def moveto(self, goal, **kw):
        """Free RRT to ``goal`` (a config, or a tcp pose IK'd via the bound tcp),
        carrying whatever is held. Returns the planned segment ``MotionData``, or
        ``None`` if it -- OR an earlier step -- couldn't plan. CHECK the return and
        stop on ``None``: there is no fluent chaining and no silent continuation.
        ``.result`` concatenates the successful segments."""
        if self._failed:
            return None
        diag = self._step_diag('moveto', kw)
        return self._record(self.arm.moveto(
            goal, collider=self.collider, constraints=self.constraints,
            tcp=self.tcp, start_qs=self._start, ee_qpos=self._ee_qpos,
            diag=diag, **kw), diag)

    def linear(self, goal_pos, goal_rotmat, **kw):
        """Straight cartesian move of the bound tcp to ``(goal_pos, goal_rotmat)``
        (the mating / insertion leg). Returns the segment ``MotionData`` or
        ``None`` (same contract as :meth:`moveto`)."""
        if self._failed:
            return None
        diag = self._step_diag('linear', kw)
        return self._record(self.arm.insert(
            goal_pos, goal_rotmat, collider=self.collider, tcp=self.tcp,
            start_qs=self._start, constraints=self.constraints,
            ee_qpos=self._ee_qpos, diag=diag, **kw), diag)

    # ---- state steps (mutate the held state; return nothing) -----------------
    def hold(self, obj, *, qpos=None, opening=None, exclude=False):
        """Take ``obj`` into the end effector at closed config ``qpos`` (or a jaw
        ``opening``). ``exclude`` keeps it out of collision (clearance enforced by
        a constraint instead). Subsequent legs carry it. A no-op if a prior step
        failed."""
        if self._failed:
            return
        # pose the arm at the CURRENT config so the mount captures the right
        # gripper<->object relative transform (the object must already sit at its
        # grasp pose, e.g. resting at the pick).
        self.arm.body.fk(qs=self._start)
        if qpos is None:
            self._ee.set_opening(opening) if opening is not None else \
                self._ee.close()
            qpos = np.asarray(self._ee.qs, dtype=np.float32).copy()
        self.arm.hold(obj, qpos, collider=self.collider, exclude=exclude)
        self._ee_qpos = np.asarray(qpos, dtype=np.float32).copy()
        self._held = obj

    def release(self, obj):
        """Drop ``obj`` (open + release); subsequent legs are empty-handed. A no-op
        if a prior step failed."""
        if self._failed:
            return
        self.arm.release(obj, collider=self.collider)
        self._ee.open()
        self._ee_qpos = np.asarray(self._ee.qs, dtype=np.float32).copy()
        self._held = None

    # ---- result --------------------------------------------------------------
    @property
    def result(self):
        """The accumulated MotionData, or None if any step failed / nothing planned."""
        if self._failed or not self._segs:
            return None
        motion = self._segs[0]
        for seg in self._segs[1:]:
            motion = motion + seg
        return motion

    @property
    def last_qs(self):
        """The config the recipe currently ends at (e.g. a hold config to freeze
        this arm while another plans), or None if failed."""
        return None if self._failed else np.asarray(self._start, dtype=np.float32)

    @property
    def segments(self):
        """The per-step motion segments (a list of MotionData, one per ``moveto`` /
        ``linear``), for a viewer that animates phase by phase; ``result`` is their
        concatenation. Empty if the Recipe failed."""
        return [] if self._failed else list(self._segs)

    @property
    def failure(self):
        """The failed step's :class:`~wrs.motion.core.diagnosis.Diagnosis`
        (``step`` names the step, ``stage``/``detail``/``counts`` say where the
        primitive gave up), or None while nothing has failed. The step returns
        stay the control flow; this is the explanation for a caller that
        REPAIRS -- e.g. 'ik' with zero candidates wants another pose, 'rrt'
        wants a larger budget."""
        return self._failure

    def _step_diag(self, verb, kw):
        """The Diagnosis for the step about to plan: the caller's own (popped
        from ``kw`` so the verb isn't passed ``diag`` twice) or a fresh one;
        either way labeled with the step index + verb."""
        diag = kw.pop('diag', None)
        if diag is None:
            diag = Diagnosis()
        if diag.step is None:
            diag.step = f'{len(self._segs)}:{verb}'
        return diag

    def _record(self, seg, diag=None):
        """Record a planned leg (fill held-object poses, advance the start config)
        and RETURN the recorded MotionData; on a failed leg set ``_failed`` (and
        keep the step's ``diag`` as ``failure``) and return None."""
        if seg is None:
            self._failed = True
            self._failure = diag
            return None
        qlist = seg.robot_qpos_list
        if self._held is not None:           # held object follows the gripper (FK)
            obj_list = []
            for q in qlist:
                self.arm.body.fk(qs=q)
                obj_list.append(np.asarray(self._held.tf, dtype=np.float32).copy())
        else:
            obj_list = [None] * len(qlist)
            self.arm.body.fk(qs=qlist[-1])   # leave the arm at the segment end so
                                             # a following grasp mounts correctly
        recorded = MotionData(qlist, seg.ee_qpos_list, obj_list)
        self._segs.append(recorded)
        self._start = np.asarray(qlist[-1], dtype=np.float64)
        return recorded
