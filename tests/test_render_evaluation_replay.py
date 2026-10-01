import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from render_evaluation_replay import load_frames, vehicle_polygon


class ReplayRenderingTests(unittest.TestCase):
    def test_vehicle_heading_and_front_bumper(self):
        north = vehicle_polygon({"x": "0", "y": "0", "angle": "0"})
        self.assertEqual(north, [(.9, 0), (.9, -5), (-.9, -5), (-.9, 0)])
        east = vehicle_polygon({"x": "0", "y": "0", "angle": "90"})
        self.assertAlmostEqual(east[1][0], -5)
        self.assertAlmostEqual(east[1][1], -.9)

    def test_frame_sampling_and_signal_alignment(self):
        with tempfile.TemporaryDirectory(prefix="replay_render_test_") as temporary:
            directory = Path(temporary)
            with gzip.open(directory / "fcd.xml.gz", "wt") as stream:
                stream.write('<fcd-export><timestep time="0"><vehicle id="a" x="0" y="0"/></timestep>'
                             '<timestep time="1"><vehicle id="a" x="1" y="0"/></timestep>'
                             '<timestep time="2"><vehicle id="a" x="2" y="0"/></timestep></fcd-export>')
            with gzip.open(directory / "tls_states.xml.gz", "wt") as stream:
                stream.write('<tlsStates><tlsState time="0" id="TL" phase="0" state="G"/>'
                             '<tlsState time="1" id="TL" phase="1" state="y"/>'
                             '<tlsState time="2" id="TL" phase="2" state="r"/></tlsStates>')
            frames = load_frames(directory, 0, 2, 2)
            self.assertEqual([frame["time"] for frame in frames], [0, 2])
            self.assertEqual([frame["signal"]["state"] for frame in frames], ["G", "r"])
            self.assertEqual(frames[0]["vehicles"][0]["x"], "0")
            self.assertEqual(frames[1]["vehicles"][0]["x"], "2")
            with self.assertRaisesRegex(ValueError, "No recorded frames"):
                load_frames(directory, 3, 4, 1)


if __name__ == "__main__":
    unittest.main()
