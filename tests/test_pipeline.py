import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pipeline


class PipelineTest(unittest.TestCase):
    def test_next_run_dir_uses_next_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "assets" / "runs"
            (runs_dir / "run_1").mkdir(parents=True)
            (runs_dir / "run_7").mkdir()
            (runs_dir / "run_draft").mkdir()

            self.assertEqual(pipeline.next_run_dir(runs_dir), runs_dir / "run_8")

    def test_resolve_cli_paths_defaults_to_run_debug_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "assets" / "runs"
            (runs_dir / "run_1").mkdir(parents=True)
            args = SimpleNamespace(
                runs_dir=runs_dir,
                run_dir=None,
                profile_json=None,
                pipe_3d_json=None,
                obj=None,
                debug_dir=None,
            )

            resolved = pipeline.resolve_cli_paths(args)

            self.assertEqual(resolved.run_dir, runs_dir / "run_2")
            self.assertEqual(resolved.profile_json, runs_dir / "run_2" / "points.json")
            self.assertEqual(resolved.pipe_3d_json, runs_dir / "run_2" / "pipe_3d.json")
            self.assertEqual(resolved.obj, runs_dir / "run_2" / "pipe.obj")
            self.assertEqual(resolved.debug_dir, runs_dir / "run_2" / "debug-pipeline")

    def test_resolve_cli_paths_keeps_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            args = SimpleNamespace(
                runs_dir=tmp / "runs",
                run_dir=tmp / "custom-run",
                profile_json=tmp / "profile.json",
                pipe_3d_json=tmp / "pipe.json",
                obj=tmp / "pipe.obj",
                debug_dir=tmp / "debug",
            )

            resolved = pipeline.resolve_cli_paths(args)

            self.assertEqual(resolved.run_dir, tmp / "custom-run")
            self.assertEqual(resolved.profile_json, tmp / "profile.json")
            self.assertEqual(resolved.pipe_3d_json, tmp / "pipe.json")
            self.assertEqual(resolved.obj, tmp / "pipe.obj")
            self.assertEqual(resolved.debug_dir, tmp / "debug")

    def test_full_pipeline_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary = pipeline.run_pipeline(
                profile_image=[Path("assets/pipe.png")],
                plan_image=Path("assets/img.png"),
                profile_json=tmp / "points.json",
                pipe_3d_json=tmp / "pipe_3d.json",
                obj_output=tmp / "pipe.obj",
                diameter_ft=0.5,
                debug_dir=tmp / "debug",
                profile_epsilon=8.0,
                sample_ft=10.0,
                plan_simplify_px=2.0,
                obj_segments=8,
                cap_ends=True,
                object_name="pipeline_pipe",
            )

            self.assertTrue((tmp / "points.json").exists())
            self.assertTrue((tmp / "pipe_3d.json").exists())
            self.assertTrue((tmp / "pipe.obj").exists())
            self.assertTrue((tmp / "debug" / "summary.json").exists())

        self.assertGreaterEqual(summary["profile_points"], 6)
        self.assertGreaterEqual(summary["pipe_3d_points"], 20)
        self.assertGreater(summary["obj_vertices"], 0)
        self.assertGreater(summary["obj_faces"], 0)

    def test_use_gemini_plan_preprocesses_plan_before_3d_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            expected_gemini_plan = tmp / "debug" / "gemini" / "plan.png"
            gemini_highlighter = mock.Mock()
            gemini_highlighter.highlight_force_main.return_value = True

            with (
                mock.patch.object(pipeline.profile_parser, "parse_profiles", return_value={"points": [{"station_ft": 0}]}),
                mock.patch.dict("sys.modules", {"gemini_highlighter": gemini_highlighter}),
                mock.patch.object(pipeline.plan_to_3d, "build_pipe_3d", return_value={"points": [{"x": 1}]}) as build_pipe_3d,
                mock.patch.object(pipeline.pipe_json_to_obj, "convert_json_to_obj", return_value=(8, 6)),
            ):
                summary = pipeline.run_pipeline(
                    profile_image=[Path("assets/profile.png")],
                    plan_image=Path("assets/top.png"),
                    profile_json=tmp / "profile_points.json",
                    pipe_3d_json=tmp / "pipe_3d.json",
                    obj_output=tmp / "pipe.obj",
                    diameter_ft=0.5,
                    debug_dir=tmp / "debug",
                    profile_epsilon=8.0,
                    sample_ft=10.0,
                    plan_simplify_px=2.0,
                    obj_segments=8,
                    cap_ends=True,
                    object_name="pipeline_pipe",
                    use_gemini_plan=True,
                )

            gemini_highlighter.highlight_force_main.assert_called_once_with(Path("assets/top.png"), expected_gemini_plan)
            build_pipe_3d.assert_called_once()
            self.assertEqual(build_pipe_3d.call_args.args[0], expected_gemini_plan)
            self.assertEqual(summary["plan_image"], "assets/top.png")
            self.assertEqual(summary["plan_image_used"], str(expected_gemini_plan))
            self.assertTrue(summary["gemini"]["plan_enabled"])


if __name__ == "__main__":
    unittest.main()
