"""Drive the planner through the Claude Code CLI -- no API key needed.

``claude -p`` (headless print mode) answers a prompt using the developer's
existing Claude Code SUBSCRIPTION login, so the agent loop can run on a
machine with no ``ANTHROPIC_API_KEY``. This adapter speaks just enough of the
``anthropic`` client surface for :func:`wrs.agent.harness.solve` --
``client.messages.create(model=, system=, messages=,...)`` returning an object
whose ``.content`` holds text blocks -- and nothing more.

``claude -p`` is single-shot, so the conversation is FLATTENED into one
prompt per call ([USER]/[ASSISTANT] transcript + the system text via
``--system-prompt``). That re-sends the history each round -- fine for the
handful of revision rounds the loop runs.

Auth: log in once interactively (``claude login``), or mint a long-lived
headless token with ``claude setup-token``. If the subprocess fails, the
error text is raised as-is -- an expired login reads exactly like the CLI
prints it.
"""
import subprocess
import types


class _Messages:
    def __init__(self, executable, timeout):
        self._executable = executable
        self._timeout = timeout

    def create(self, *, model, system='', messages=(), **_):
        transcript = [f"[{m['role'].upper()}]\n{m['content']}"
                      for m in messages]
        prompt = '\n\n'.join(transcript)
        cmd = [self._executable, '-p', '--model', model,
               '--no-session-persistence']
        if system:
            cmd += ['--system-prompt', system]
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=self._timeout)
        if proc.returncode != 0:
            raise RuntimeError(f'claude -p failed: '
                               f'{proc.stderr.strip() or proc.stdout.strip()}')
        block = types.SimpleNamespace(type='text', text=proc.stdout)
        return types.SimpleNamespace(content=[block])


class ClaudeCodeClient:
    """``anthropic``-shaped client backed by ``claude -p`` (subscription)."""

    def __init__(self, executable='claude', timeout=600):
        self.messages = _Messages(executable, timeout)
