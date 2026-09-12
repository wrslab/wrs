"""The skill catalog -- the library's manipulation vocabulary as DATA.

The library's accumulated robotics know-how is exposed to a task-decomposition
layer (an LLM planner, a TAMP search, a script generator) as a small set of
ATOMIC skills. Two kinds, both load-bearing:

  * **actions** plan or mutate -- they move the arm or change the held state.
  * **queries** produce information -- feasible grasps, stable placements,
    reachability. Decomposition lives or dies on these: the hard part of a
    multi-step task is the geometric constraint that couples the steps (a
    grasp must be feasible at BOTH the pick and the place), and the queries
    are exactly that reasoning, precomputed and callable.

Deliberately FEW entries: an atom per orthogonal verb / query, no
combinatorial family of named compositions (see the Recipe docstring for the
argument) -- ``pick_and_place`` is included as the one canonical composition,
labeled as such. Compositions are written as Recipes (data), not added here.

This module is a PURE LITERAL (no imports, no computation): the manifest
generator (``tools/gen_skill_manifest.py``) reads it statically -- like
``gen_api_index.py``, no import side effects -- validates every ``target``
against the source tree (a renamed function fails the build, so the catalog
cannot go stale silently), attaches the live signature + docstring, and
renders ``docs/SKILLS.md`` (human) and ``docs/skills.json`` (machine).

Entry fields:
  name        the skill's name in the planner's vocabulary
  kind        'motion' / 'state' / 'query' / 'composed' / 'setup'
  target      dotted path to the implementing callable (validated)
  recipe_verb the Recipe step exposing it, when one does
  summary     one line, written for the PLANNER audience
  pre         preconditions the caller must establish
  post        what holds after success
  returns     the success value
  failures    Diagnosis stage -> the repair it calls for (motion skills;
              see wrs.motion.core.diagnosis for the stage vocabulary)
"""

SKILLS = (
    # ---- setup ---------------------------------------------------------------
    {
        'name': 'build_collider',
        'kind': 'setup',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.build_collider',
        'summary': 'Build the collision world every planning skill runs in: '
                   'robot (+ mounted gripper) + static fixtures + obstacles.',
        'pre': ['robot constructed; end effector mounted',
                'the object about to be PICKED is NOT included (it is grasped, '
                'not an obstacle)'],
        'post': ['a compiled collider, passed explicitly (collider=) to every '
                 'planning call'],
        'returns': 'MJCollider',
        'failures': {},
    },
    # ---- motion actions ------------------------------------------------------
    {
        'name': 'moveto',
        'kind': 'motion',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.moveto',
        'recipe_verb': 'moveto',
        'summary': 'Free collision-planned travel (RRT) to a config or a tcp '
                   'pose, carrying whatever is held.',
        'pre': ['collider built; start config valid',
                'pose goals need a tcp (the working frame)'],
        'post': ['arm at the goal; held object carried collision-checked'],
        'returns': 'MotionData, or None (why: diag= / recipe.failure)',
        'failures': {
            'ik': 'zero candidates: the pose is out of reach -- change the pose '
                  'or the base; all invalid: it is blocked -- change the world '
                  'or the grasp',
            'start_invalid': 'the world is mis-set-up (start already colliding)',
            'goal_invalid': 'the explicit goal config collides -- pick another',
            'rrt': 'no path in budget -- retry with a larger budget or another '
                   'goal IK branch',
        },
    },
    {
        'name': 'linear',
        'kind': 'motion',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.insert',
        'recipe_verb': 'linear',
        'summary': 'Straight cartesian leg of the tcp (no RRT) -- the mating / '
                   'insertion / press move.',
        'pre': ['collider built; the line start is the current tcp pose of '
                'start_qs (or given explicitly)'],
        'post': ['tcp moved straight to the goal pose'],
        'returns': 'MotionData, or None (why: diag= / recipe.failure)',
        'failures': {
            'cartesian': 'ik broke: the line exits the workspace -- shorten it '
                         'or reorient; blocked: an obstacle crosses the line -- '
                         'clear it or approach from elsewhere',
        },
    },
    {
        'name': 'approach',
        'kind': 'motion',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.approach',
        'summary': 'RRT to a pre-grasp then a straight descent along the '
                   'approach axis -- how every pick and place reaches its '
                   'target.',
        'pre': ['collider built; grasp/goal pose chosen (e.g. from '
                'reason_grasps)'],
        'post': ['tcp at the grasp pose, having come in straight'],
        'returns': 'MotionData, or None (why: diag=)',
        'failures': {
            'ik': 'pre-grasp unreachable or blocked -- try another grasp',
            'rrt': 'travel found no path -- larger budget or another grasp',
            'cartesian': 'descent broke or is blocked -- another approach '
                         'direction or grasp',
        },
    },
    {
        'name': 'depart',
        'kind': 'motion',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.depart',
        'summary': 'Straight retreat along the approach axis (back out the way '
                   'the hand came in), optionally RRT-parking after.',
        'pre': ['at a grasp/contact pose (start_qs required)'],
        'post': ['tcp retreated depart_distance; optionally parked at end_qs'],
        'returns': 'MotionData, or None (why: diag=)',
        'failures': {
            'cartesian': 'retreat blocked -- another depart direction',
            'rrt': 'park leg found no path -- larger budget or another park',
        },
    },
    # ---- state actions -------------------------------------------------------
    {
        'name': 'hold',
        'kind': 'state',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.hold',
        'recipe_verb': 'hold',
        'summary': 'Take the object into the end effector at the current '
                   'config; subsequent motion carries it collision-checked.',
        'pre': ['arm AT the grasp config (plan there first)',
                'object resting at its grasp pose'],
        'post': ['object mounted on the gripper; collider refreshed'],
        'returns': 'None (state mutation)',
        'failures': {},
    },
    {
        'name': 'release',
        'kind': 'state',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.release',
        'recipe_verb': 'release',
        'summary': 'Drop the held object (open + unmount); subsequent motion '
                   'is empty-handed.',
        'pre': ['object currently held; arm at the place pose'],
        'post': ['object free at its current pose; collider refreshed'],
        'returns': 'None (state mutation)',
        'failures': {},
    },
    # ---- queries -------------------------------------------------------------
    {
        'name': 'reachable',
        'kind': 'query',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.reachable',
        'summary': 'Cheap feasibility gate: the nearest collision-free IK for '
                   'a pose, WITHOUT motion planning -- vet every key pose of a '
                   'plan before paying for any leg.',
        'pre': ['collider built'],
        'post': [],
        'returns': 'a config, or None (diag counts split unreachable vs '
                   'blocked)',
        'failures': {},
    },
    {
        'name': 'reason_grasps',
        'kind': 'query',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.reason_grasps',
        'summary': 'Which grasps are reachable + collision-free at ALL the '
                   'given object poses -- THE coupling constraint of '
                   'pick-then-place; an empty result is what calls for a '
                   'regrasp.',
        'pre': ['object-local grasps generated (generate_grasps) or loaded',
                'collider built'],
        'post': [],
        'returns': '{gid: [config at each pose, ...]}; empty -> no single '
                   'grasp works everywhere -- insert an intermediate placement '
                   '(regrasp) or add grasps',
        'failures': {},
    },
    {
        'name': 'generate_grasps',
        'kind': 'query',
        'target': 'wrs.grasp.antipodal.antipodal',
        'summary': 'Generate object-local antipodal (2-point pinch) grasp '
                   'candidates for a gripper on a mesh.',
        'pre': ['gripper model; object SceneObject with mesh'],
        'post': [],
        'returns': 'list of Grasp (object-local); empty -> loosen density / '
                   'tolerances, or the object does not fit the jaw',
        'failures': {},
    },
    {
        'name': 'stable_placements',
        'kind': 'query',
        'target': 'wrs.grasp.placement.compute_stable_poses',
        'summary': 'The poses an object can REST in -- the intermediate '
                   'placements a regrasp routes through.',
        'pre': ['object mesh (+ optional center of mass)'],
        'post': [],
        'returns': 'stable resting poses with support facets',
        'failures': {},
    },
    # ---- the canonical composition -------------------------------------------
    {
        'name': 'pick_and_place',
        'kind': 'composed',
        'target': 'wrs.manipulation.arm.SingleArmManipulation.pick_and_place',
        'summary': 'home -> pick -> lift -> transfer -> place -> release -> '
                   'retreat, choosing a grasp feasible at BOTH ends '
                   '(reason_grasps inside). The one canonical composition; '
                   'other shapes are written as Recipes, not added as skills.',
        'pre': ['grasps generated; pick/place poses known; collider WITHOUT '
                'the object'],
        'post': ['object resting at the place pose; arm retreated'],
        'returns': 'MotionData, or None (no common grasp fully planned -- '
                   'regrasp via stable_placements, or add grasps)',
        'failures': {},
    },
)
