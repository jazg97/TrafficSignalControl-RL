import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from presentation_replay import collect_metrics


class PresentationMetricsTests(unittest.TestCase):
    def test_delayed_departure_unfinished_trip_and_outside_crop_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'environment.net.xml').write_text(
                '<net><junction id="TL" x="0" y="0"/><junction id="N" x="0" y="500"/>'
                '<edge id="in" from="N" to="TL"><lane id="in_0"/></edge></net>')
            (directory / 'episode_routes.rou.xml').write_text(
                '<routes><vehicle id="a" depart="0"/><vehicle id="b" depart="1"/></routes>')
            with gzip.open(directory / 'tripinfo.xml.gz', 'wt') as stream:
                stream.write('<tripinfos><tripinfo id="a" depart="2" arrival="60"/>'
                             '<tripinfo id="b" depart="3" arrival="-1"/></tripinfos>')
            with gzip.open(directory / 'fcd.xml.gz', 'wt') as stream:
                stream.write('<fcd-export><timestep time="1"/>'
                             '<timestep time="3"><vehicle id="a" lane="in_0" speed="0" x="0" y="400"/>'
                             '<vehicle id="b" lane="in_0" speed="0.10" x="0" y="300"/></timestep>'
                             '<timestep time="62"><vehicle id="b" lane="out_0" speed="0"/></timestep></fcd-export>')
            rows, scheduled = collect_metrics(directory)
            self.assertEqual(scheduled, [0, 1])
            self.assertEqual((rows[0]['scheduled'], rows[0]['inserted']), (2, 0))
            self.assertEqual((rows[1]['stopped_N'], rows[1]['stopped_incoming']), (1, 1))
            self.assertEqual(rows[2]['completed'], 1)
            self.assertEqual(rows[2]['inserted_last_60s'], 1)
            self.assertEqual(rows[2]['stopped_incoming'], 0)


if __name__ == '__main__':
    unittest.main()
