"""Task decomposition: an LLM writes Recipe code against the skill catalog.

The harness (``solve``) closes the loop the skill layer was built for: the
skill manifest (``docs/SKILLS.md``) tells the model WHAT it can call, the
Diagnosis channel (``recipe.failure``) tells it WHY an attempt failed, and the
model revises. Kept out of ``wrs/__init__`` on purpose -- the ``anthropic``
dependency is optional (``pip install wrs[agent]``) and nothing else in the
library needs it.
"""
from wrs.agent.cli_client import ClaudeCodeClient
from wrs.agent.executor import run_script
from wrs.agent.harness import default_client, solve
