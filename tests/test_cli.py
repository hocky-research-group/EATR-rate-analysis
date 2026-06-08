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


class CliTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_rates_cli_bootstrap_threads_writes_ci_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            colvar1 = tmp_path / "traj1.colvar"
            colvar2 = tmp_path / "traj2.colvar"
            colvar3 = tmp_path / "traj3.colvar"
            output = tmp_path / "rates_bootstrap.json"
            repo_root = Path(__file__).resolve().parents[1]

            _write_colvar(colvar1, [(0, 0, 0.1), (1, 0, 0.15), (2, 0, 0.2), (3, 0, 0.25)])
            _write_colvar(colvar2, [(0, 0, 0.2), (1, 0, 0.25), (2, 0, 0.3), (3, 0, 0.35)])
            _write_colvar(colvar3, [(0, 0, 0.3), (1, 0, 0.35), (2, 0, 0.4), (3, 0, 0.45)])

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eatr_rates",
                    "-i",
                    str(colvar1),
                    str(colvar2),
                    str(colvar3),
                    "-e",
                    "-b",
                    "--numboots",
                    "6",
                    "--threads",
                    "2",
                    "-q",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("EATR MLE ln k", payload)
            self.assertIn("EATR MLE gamma", payload)
            self.assertIn("EATR MLE ln k CI", payload)
            self.assertIn("EATR MLE gamma CI", payload)
            self.assertIn("EATR MLE CDF plot", payload)
            self.assertEqual(sorted(payload["EATR MLE CDF plot"]), ["ecdf", "fit", "time"])
            self.assertEqual(len(payload["EATR MLE CDF plot"]["time"]), len(payload["EATR MLE CDF plot"]["ecdf"]))
            self.assertEqual(len(payload["EATR MLE CDF plot"]["time"]), len(payload["EATR MLE CDF plot"]["fit"]))

    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_rates_cli_writes_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            colvar1 = tmp_path / "traj1.colvar"
            colvar2 = tmp_path / "traj2.colvar"
            output = tmp_path / "rates.json"
            repo_root = Path(__file__).resolve().parents[1]

            _write_colvar(colvar1, [(0, 0, 0.5), (1, 0, 0.5), (2, 0, 0.5)])
            _write_colvar(colvar2, [(0, 0, 0.2), (1, 0, 0.2), (2, 0, 0.2)])

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eatr_rates",
                    "-i",
                    str(colvar1),
                    str(colvar2),
                    "-m",
                    "-q",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("iMetaD MLE ln k", payload)
            self.assertIn("iMetaD MLE KS stat", payload)

    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_check_order_cli_writes_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "order.dat"
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "check_order",
                    "-i",
                    "a.colvar",
                    "b.colvar",
                    "-l",
                    "a.log",
                    "b.log",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["a.colvar | a.log", "b.colvar | b.log"],
            )

    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_flooding_cli_bootstrap_threads_writes_uncertainty_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_root = Path(__file__).resolve().parents[1]
            output = tmp_path / "flooding_bootstrap.json"

            set1_traj1 = tmp_path / "set1_traj1.colvar"
            set1_traj2 = tmp_path / "set1_traj2.colvar"
            set1_traj3 = tmp_path / "set1_traj3.colvar"
            set2_traj1 = tmp_path / "set2_traj1.colvar"
            set2_traj2 = tmp_path / "set2_traj2.colvar"
            set2_traj3 = tmp_path / "set2_traj3.colvar"
            _write_colvar(set1_traj1, [(0, 0, 0.1), (1, 0, 0.2), (2, 0, 0.3), (3, 0, 0.4)])
            _write_colvar(set1_traj2, [(0, 0, 0.12), (1, 0, 0.22), (2, 0, 0.32), (3, 0, 0.42)])
            _write_colvar(set1_traj3, [(0, 0, 0.14), (1, 0, 0.24), (2, 0, 0.34), (3, 0, 0.44)])
            _write_colvar(set2_traj1, [(0, 0, 0.4), (1, 0, 0.5), (2, 0, 0.6), (3, 0, 0.7)])
            _write_colvar(set2_traj2, [(0, 0, 0.42), (1, 0, 0.52), (2, 0, 0.62), (3, 0, 0.72)])
            _write_colvar(set2_traj3, [(0, 0, 0.44), (1, 0, 0.54), (2, 0, 0.64), (3, 0, 0.74)])

            log_paths = []
            for index in range(6):
                log_path = tmp_path / f"log{index}.log"
                log_path.write_text("#! SET COMMIT(T)ED TO BASIN 1\n", encoding="utf-8")
                log_paths.append(log_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eatr_rates.rates_eatr_opes",
                    "-i",
                    str(set1_traj1),
                    str(set1_traj2),
                    str(set1_traj3),
                    "--barrier",
                    "1",
                    "-i",
                    str(set2_traj1),
                    str(set2_traj2),
                    str(set2_traj3),
                    "--barrier",
                    "2",
                    "--logfiles",
                    str(log_paths[0]),
                    str(log_paths[1]),
                    str(log_paths[2]),
                    "--logfiles",
                    str(log_paths[3]),
                    str(log_paths[4]),
                    str(log_paths[5]),
                    "--beta",
                    "1.0",
                    "--tcol",
                    "0",
                    "--vcol",
                    "2",
                    "--bootstrap",
                    "--numboots",
                    "6",
                    "--threads",
                    "2",
                    "-q",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("bootstrap_logk0_std", payload)
            self.assertIn("bootstrap_gamma_std", payload)
            self.assertEqual(len(payload["bootstrap_iterations"]), 6)
            self.assertTrue((tmp_path / "flooding_bootstrap_observed_rate.png").exists())
            self.assertTrue((tmp_path / "flooding_bootstrap_ln_kobs_vs_acceleration.png").exists())
            self.assertTrue((tmp_path / "flooding_bootstrap_diagnostics.png").exists())

    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_flooding_cli_writes_json_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_root = Path(__file__).resolve().parents[1]
            output = tmp_path / "flooding.json"

            set1_traj1 = tmp_path / "set1_traj1.colvar"
            set1_traj2 = tmp_path / "set1_traj2.colvar"
            set2_traj1 = tmp_path / "set2_traj1.colvar"
            set2_traj2 = tmp_path / "set2_traj2.colvar"
            _write_colvar(set1_traj1, [(0, 0, 0.1), (1, 0, 0.2), (2, 0, 0.3)])
            _write_colvar(set1_traj2, [(0, 0, 0.2), (1, 0, 0.3), (2, 0, 0.4)])
            _write_colvar(set2_traj1, [(0, 0, 0.4), (1, 0, 0.5), (2, 0, 0.6)])
            _write_colvar(set2_traj2, [(0, 0, 0.5), (1, 0, 0.6), (2, 0, 0.7)])

            log_paths = []
            for index in range(4):
                log_path = tmp_path / f"log{index}.log"
                log_path.write_text("#! SET COMMIT(T)ED TO BASIN 1\n", encoding="utf-8")
                log_paths.append(log_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eatr_rates.rates_eatr_opes",
                    "-i",
                    str(set1_traj1),
                    str(set1_traj2),
                    "--barrier",
                    "1",
                    "-i",
                    str(set2_traj1),
                    str(set2_traj2),
                    "--barrier",
                    "2",
                    "--logfiles",
                    str(log_paths[0]),
                    str(log_paths[1]),
                    "--logfiles",
                    str(log_paths[2]),
                    str(log_paths[3]),
                    "--beta",
                    "1.0",
                    "--tcol",
                    "0",
                    "--vcol",
                    "2",
                    "-q",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("gamma", payload)
            self.assertIn("logk0", payload)
            self.assertEqual(len(payload["set_reports"]), 2)
            self.assertIn("flooding_diagnostics", payload)
            self.assertTrue((tmp_path / "flooding_observed_rate.png").exists())
            self.assertTrue((tmp_path / "flooding_ln_kobs_vs_acceleration.png").exists())
            self.assertTrue((tmp_path / "flooding_diagnostics.png").exists())

    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_plot_results_cli_writes_regular_series_and_cdf_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_root = Path(__file__).resolve().parents[1]
            input1 = tmp_path / "pace_1ps.json"
            input2 = tmp_path / "pace_10ps.json"
            output = tmp_path / "regular.png"
            cdf_output = tmp_path / "regular_cdf.png"

            input1.write_text(
                json.dumps(
                    {
                        "EATR MLE ln k": 1.0,
                        "EATR MLE gamma": 0.2,
                        "EATR MLE ln k std": 0.1,
                        "EATR MLE gamma std": 0.02,
                        "EATR MLE CDF plot": {
                            "time": [1.0, 2.0, 3.0],
                            "ecdf": [0.5, 0.75, 1.0],
                            "fit": [0.4, 0.7, 0.95],
                        },
                    }
                ),
                encoding="utf-8",
            )
            input2.write_text(
                json.dumps(
                    {
                        "EATR MLE ln k": 2.0,
                        "EATR MLE gamma": 0.4,
                        "EATR MLE ln k std": 0.2,
                        "EATR MLE gamma std": 0.03,
                        "EATR MLE CDF plot": {
                            "time": [1.5, 2.5, 3.5],
                            "ecdf": [0.4, 0.7, 1.0],
                            "fit": [0.35, 0.68, 0.98],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eatr_rates.plot_results",
                    "regular-series",
                    "-i",
                    str(input1),
                    str(input2),
                    "--labels",
                    "pace 1 ps",
                    "pace 10 ps",
                    "--method",
                    "eatr-mle",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertTrue(cdf_output.exists())
