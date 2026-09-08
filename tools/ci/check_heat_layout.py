"""Exercise the read-only heat-layout CLI against a committed container."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    engine, container = sys.argv[1:3]
    with tempfile.TemporaryDirectory(prefix="soma-heat-layout-") as work:
        heat = Path(work) / "heat.json"
        cells = [dict(layer=1, expert=e, count=10, decayed=10.0) for e in (1, 3)]

        def run(*args):
            return subprocess.run([engine, "heat-layout", container, "--heat", str(heat),
                                   "--json", *args], capture_output=True, text=True)

        heat.write_text(json.dumps(dict(experts=cells)), encoding="utf-8")
        result = run()
        assert result.returncode == 0, result.stderr
        base = json.loads(result.stdout)
        assert base["pinned_experts"] == 2
        assert base["runs"] == 2  # experts 1 and 3 have expert 2 between them
        assert base["runs_ideal_per_layer"] == 1
        assert "cache capacity not modeled" in base["selection"]

        heat.write_text(json.dumps(dict(experts=[cells[0], *cells])), encoding="utf-8")
        duplicate = run()
        assert duplicate.returncode == 0, duplicate.stderr
        assert json.loads(duplicate.stdout) == base, "duplicate changed layout statistics"

        one = run("--pin", str(base["pinned_bytes"] // 2))
        assert one.returncode == 0, one.stderr
        assert json.loads(one.stdout)["pinned_experts"] == 1
        for args in [("--pin", "-1"), ("--pin", "nope"), ("--pin", "12x"),
                     ("--pin", "18446744073709551616"), ("--pin",), ("--unknown",)]:
            assert run(*args).returncode != 0, f"accepted invalid arguments: {args}"
        assert run("--pin", "1").returncode != 0
        heat.write_text('{"experts":[{"layer":"bad","expert":1}]}', encoding="utf-8")
        assert run().returncode != 0
    print("heat-layout: CLI, duplicate, budget, and layout checks passed")


if __name__ == "__main__":
    main()
