"""The write -> run -> diagnose -> revise loop: an LLM plans with the skills.

``solve`` gives a model the task, a description of a prepared scene, and the
skill manifest (``docs/SKILLS.md`` -- the same catalog a human reads); the
model answers with ONE Python script that composes the skills; the script runs
in simulation (:mod:`wrs.agent.executor`); on failure the model is shown WHY
-- the stamped :class:`~wrs.motion.core.diagnosis.Diagnosis`, the traceback,
or a failed success check -- and revises. The loop is the consumer the
Diagnosis channel and the manifest were built for: 'ik' with zero candidates
reads "move the target", 'rrt' reads "retry bigger", an empty grasp set reads
"regrasp".

Scenes are FACTORIES (``build_scene() -> (namespace, description)``), not
values: scene objects mutate as a plan runs (fk moves the robot, hold mounts
the object), so every attempt plans against a fresh build -- no state bleed
between revisions.

The LLM client is passed explicitly like every other collaborator (default:
``anthropic.Anthropic()``, resolving credentials from the environment) -- so
tests inject a fake and pay nothing.
"""
import os
import re

from wrs.agent.executor import run_script

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MANIFEST = os.path.join(_ROOT, 'docs', 'SKILLS.md')

_SYSTEM = """\
You are a robot manipulation planner working with the `wrs` robotics library.
You are given a TASK and a SCENE: Python variables that already exist in the
namespace your code will run in. Reply with exactly ONE ```python code block
and nothing else.

Rules for the script:
- Use ONLY the scene variables described in the user message (plus `np`,
  which is always provided). Do not import anything.
- Compose behavior from the skills listed below: prefer the Recipe verbs for
  multi-step motion, and use the QUERY skills to reason before planning
  (e.g. gate key poses with `reachable` before paying for a `moveto`).
- Every motion step returns a MotionData or None. CHECK every return and stop
  on None -- no silent continuation.
- The deliverable is the variable `motion`: assign the accumulated MotionData
  on success (e.g. `motion = r.result`), or None if planning failed.
- If your plan fails you will be shown why (the step's Diagnosis: which stage
  gave up and its counts) and may revise -- pick the repair the stage calls
  for rather than retrying the same call.

{manifest}
"""


class Solution:
    """What ``solve`` produced: the final outcome plus the revision history."""

    def __init__(self):
        self.ok = False
        self.motion = None
        self.code = ''        # the last script the model wrote
        self.rounds = 0
        self.history = []     # (code, Outcome, feedback) per round

    def __repr__(self):
        return f'<Solution ok={self.ok} rounds={self.rounds}>'


def _extract_code(text):
    """The first ```python block, or the raw text as a last resort."""
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    return m.group(1) if m else text


def solve(task, build_scene, *, check=None, model='claude-opus-4-8',
          max_rounds=3, client=None, log=print):
    """Decompose ``task`` into a plan over a scene, revising on failure.

    ``build_scene()`` -> ``(namespace, description)``: the variables the
    script plans with, and the prose that tells the model what they are
    (names, robots, obstacles, target poses). Called fresh per attempt.

    ``check(outcome)`` optionally judges a planned result beyond "motion is
    not None" -- return None to accept, or a string saying what is wrong (fed
    back to the model). ``client`` is an ``anthropic``-compatible client
    (injected in tests); ``log`` receives one line per round (pass
    ``lambda *_: None`` to silence).

    Returns a :class:`Solution`; ``.ok`` says whether the last attempt
    planned fully and passed ``check``.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    with open(_MANIFEST, 'r', encoding='utf-8') as f:
        system = _SYSTEM.format(manifest=f.read())

    solution = Solution()
    namespace, description = build_scene()
    messages = [{'role': 'user',
                 'content': f'TASK: {task}\n\nSCENE:\n{description}'}]
    for round_no in range(1, max_rounds + 1):
        solution.rounds = round_no
        response = client.messages.create(
            model=model, max_tokens=16000, thinking={'type': 'adaptive'},
            system=system, messages=messages)
        text = ''.join(b.text for b in response.content if b.type == 'text')
        code = _extract_code(text)
        solution.code = code

        outcome = run_script(code, namespace)
        feedback = None if outcome.ok else outcome.describe()
        if feedback is None and check is not None:
            feedback = check(outcome)
        solution.history.append((code, outcome, feedback))
        if feedback is None:
            solution.ok = True
            solution.motion = outcome.motion
            log(f'[agent] round {round_no}: solved')
            return solution

        log(f'[agent] round {round_no}: {feedback.splitlines()[0]}')
        namespace, _ = build_scene()       # fresh scene for the next attempt
        messages.append({'role': 'assistant', 'content': text})
        messages.append({'role': 'user', 'content':
                         f'That attempt failed.\n{feedback}\n\n'
                         f'Revise the script (the scene has been rebuilt '
                         f'fresh). Reply with one ```python block.'})
    return solution
