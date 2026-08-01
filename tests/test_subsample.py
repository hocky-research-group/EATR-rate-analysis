from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


def _write_colvar(path: Path, rows) -> None:
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _make_inputs(tmp_path: Path, n: int) -> list[Path]:
    """n toy trajectories with a rising bias, distinct enough to give distinct fits."""
    files = []
    for i in range(n):
        p = tmp_path / f"traj{i}.colvar"
        rows = [(t, 0, 0.05 * (i + 1) * (t + 1)) for t in range(6)]
        _write_colvar(p, rows)
        files.append(p)
    return files


def _run(tmp_path: Path, files, extra, out: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "eatr_rates", "-i", *[str(f) for f in files],
           "-e", "--threads", "1", "-q", *extra, "-o", str(out)]
    return subprocess.run(
        cmd, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True, text=True, check=False,
    )


@unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
class SubsampleTests(unittest.TestCase):
    def test_absent_flag_leaves_output_untouched(self):
        """Without --subsample-runs there must be no 'subsample' key at all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 6)
            out = tmp_path / "r.json"
            res = _run(tmp_path, files, [], out)
            self.assertEqual(res.returncode, 0, res.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertNotIn("subsample", payload)

    def test_sweep_records_every_size_and_replicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 8)
            out = tmp_path / "r.json"
            res = _run(tmp_path, files,
                       ["--subsample-runs", "3,5", "--subsample-reps", "4", "--seed", "7"], out)
            self.assertEqual(res.returncode, 0, res.stderr)
            sub = json.loads(out.read_text(encoding="utf-8"))["subsample"]
            self.assertEqual(sub["parent_n"], 8)
            self.assertEqual(sub["sizes"], [3, 5])
            self.assertEqual(len(sub["results"]), 8)  # 2 sizes x 4 reps
            self.assertEqual(sub["summary"]["3"]["n_fits"], 4)
            self.assertEqual(sub["summary"]["5"]["n_fits"], 4)
            for rec in sub["results"]:
                self.assertEqual(len(rec["indices"]), rec["n"])
                self.assertEqual(len(set(rec["indices"])), rec["n"],
                                 "default draw must be WITHOUT replacement")

    def test_seed_makes_the_draw_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 8)
            a, b = tmp_path / "a.json", tmp_path / "b.json"
            args = ["--subsample-runs", "4", "--subsample-reps", "3", "--seed", "11"]
            self.assertEqual(_run(tmp_path, files, args, a).returncode, 0)
            self.assertEqual(_run(tmp_path, files, args, b).returncode, 0)
            ia = [r["indices"] for r in json.loads(a.read_text())["subsample"]["results"]]
            ib = [r["indices"] for r in json.loads(b.read_text())["subsample"]["results"]]
            self.assertEqual(ia, ib)
            # replicates within one run must differ from each other
            self.assertNotEqual(ia[0], ia[1])

    def test_size_equal_to_parent_collapses_to_one_replicate(self):
        """Drawing all of them without replacement is deterministic, so extra reps
        would just be identical fits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 5)
            out = tmp_path / "r.json"
            res = _run(tmp_path, files,
                       ["--subsample-runs", "5", "--subsample-reps", "6", "--seed", "3"], out)
            self.assertEqual(res.returncode, 0, res.stderr)
            sub = json.loads(out.read_text(encoding="utf-8"))["subsample"]
            self.assertEqual(len(sub["results"]), 1)
            self.assertEqual(sub["summary"]["5"]["n_fits"], 1)
            self.assertIsNone(sub["summary"]["5"]["EATR MLE gamma std"])

    def test_oversized_request_is_an_error_not_a_silent_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 5)
            out = tmp_path / "r.json"
            res = _run(tmp_path, files, ["--subsample-runs", "99"], out)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("exceeds", res.stderr + res.stdout)
            self.assertFalse(out.exists())

    def test_replace_allows_oversized_and_repeats_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 5)
            out = tmp_path / "r.json"
            res = _run(tmp_path, files,
                       ["--subsample-runs", "9", "--subsample-replace", "--seed", "5"], out)
            self.assertEqual(res.returncode, 0, res.stderr)
            sub = json.loads(out.read_text(encoding="utf-8"))["subsample"]
            self.assertTrue(sub["replace"])
            rec = sub["results"][0]
            self.assertEqual(len(rec["indices"]), 9)
            self.assertLess(len(set(rec["indices"])), 9)  # must contain repeats

    def test_rejects_degenerate_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            files = _make_inputs(tmp_path, 5)
            out = tmp_path / "r.json"
            res = _run(tmp_path, files, ["--subsample-runs", "1"], out)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("at least 2", res.stderr + res.stdout)


if __name__ == "__main__":
    unittest.main()
