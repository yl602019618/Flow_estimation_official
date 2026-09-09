from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from incflo_sphere_wake_comparison.tools.input_parser import parse_inputs, scalar
from incflo_sphere_wake_comparison.tools.parse_incflo_log import parse_log
from incflo_sphere_wake_comparison.tools.validate_inputs import validate_one


ROOT = (
    Path(__file__).resolve().parents[2]
    / "incflo_sphere_wake_comparison"
)
INPUTS = ROOT / "inputs"


class InputContractTests(unittest.TestCase):
    def test_all_frozen_inputs_are_valid(self) -> None:
        paths = sorted(INPUTS.glob("inputs.*"))
        self.assertEqual(len(paths), 8)
        reports = [validate_one(path) for path in paths]
        self.assertTrue(all(item["valid"] for item in reports), reports)

    def test_production_sphere_physics_does_not_drift_with_io(self) -> None:
        noio = parse_inputs(INPUTS / "inputs.sphere_8m_production_100_noio")
        output = parse_inputs(INPUTS / "inputs.sphere_8m_production_100_output")
        allowed = {"amr.plot_int", "amr.plot_file"}
        keys = set(noio) | set(output)
        differences = {key for key in keys if noio.get(key) != output.get(key)}
        self.assertEqual(differences, allowed)

    def test_baseline_differs_only_by_geometry_and_output_name(self) -> None:
        sphere = parse_inputs(INPUTS / "inputs.sphere_8m_production_100_noio")
        baseline = parse_inputs(INPUTS / "inputs.baseline_8m_production_100_noio")
        ignored = {
            "incflo.geometry", "sphere.internal_flow", "sphere.radius",
            "sphere.center", "amr.plot_file",
        }
        keys = (set(sphere) | set(baseline)) - ignored
        differences = {key for key in keys if sphere.get(key) != baseline.get(key)}
        self.assertEqual(differences, set())

    def test_mature_case_matches_validated_high_grid_and_output_cadence(self) -> None:
        short = parse_inputs(
            INPUTS / "inputs.sphere_8m_double_strict_high144_100_output"
        )
        mature = parse_inputs(
            INPUTS / "inputs.sphere_8m_double_strict_high144_mature_40s"
        )
        allowed = {
            "max_step", "amr.plot_int", "amr.check_int", "amr.plot_file",
            "amr.check_file",
        }
        keys = set(short) | set(mature)
        differences = {key for key in keys if short.get(key) != mature.get(key)}
        self.assertEqual(differences, allowed)
        self.assertEqual(int(scalar(mature, "max_step")), 13320)
        self.assertEqual(int(scalar(mature, "amr.plot_int")), 333)
        self.assertAlmostEqual(
            float(scalar(mature, "incflo.fixed_dt"))
            * int(scalar(mature, "amr.plot_int")),
            0.999,
        )

    def test_build_is_float_cuda_eb_and_contains_no_credentials(self) -> None:
        versions = (ROOT / "versions.env").read_text(encoding="utf-8")
        build = (ROOT / "scripts" / "build_remote.sh").read_text(encoding="utf-8")
        all_scripts = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / "scripts").glob("*.sh")
        )
        self.assertIn("PRECISION=FLOAT", versions)
        self.assertIn("CUDA_ARCH=120", versions)
        self.assertIn("USE_CUDA=TRUE", build)
        self.assertIn("USE_EB=TRUE", build)
        self.assertIn("INCFLO_PRECISION", build)
        self.assertIn("INCFLO_VARIANT", all_scripts)
        self.assertNotIn("password", all_scripts.lower())
        self.assertNotIn("sshpass", all_scripts.lower())


class LogParserTests(unittest.TestCase):
    def test_successful_log_and_time_are_parsed(self) -> None:
        input_path = INPUTS / "inputs.sphere_8m_smoke"
        log = "\n".join(
            [
                f"Step {step} starts at time {(step - 1) * 0.003:g} with dt = 0.003."
                for step in range(1, 6)
            ]
            + [
                "Time spent in InitData():    2.5",
                "Time spent in Evolve():      0.5",
            ]
        )
        timing = "\n".join(
            [
                "User time (seconds): 3.0",
                "System time (seconds): 0.2",
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:03.40",
                "Maximum resident set size (kbytes): 123456",
            ]
        )
        gpu = "2026/08/23 00:00:00.000, 0, 1024, 90\n"
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            log_path = temp / "log"
            time_path = temp / "time"
            gpu_path = temp / "gpu.csv"
            log_path.write_text(log, encoding="utf-8")
            time_path.write_text(timing, encoding="utf-8")
            gpu_path.write_text(gpu, encoding="utf-8")
            result = parse_log(
                log_path, input_path, time_path, gpu_path, 0, 252.93,
            )
        self.assertTrue(result["successful"], json.dumps(result, indent=2))
        self.assertEqual(result["completed_steps"], 5)
        self.assertAlmostEqual(result["evolve_seconds_per_step"], 0.1)
        self.assertAlmostEqual(result["gnu_time"]["wall_seconds"], 3.4)
        self.assertEqual(result["gpu"]["peak_memory_mib"], 1024.0)
        self.assertFalse(result["primary_performance_case"])
        self.assertFalse(result["performance_gate_3x"])

    def test_abort_is_not_success(self) -> None:
        input_path = INPUTS / "inputs.sphere_8m_smoke"
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log"
            log_path.write_text("AMReX::Abort::0:: solver failed to converge\n", encoding="utf-8")
            result = parse_log(log_path, input_path, None, None, None, 252.93)
        self.assertFalse(result["successful"])
        self.assertTrue(result["failure_flags"]["amrex_abort"])


if __name__ == "__main__":
    unittest.main()
