"""Self-contained SUMO trajectory and traffic-light recordings for one episode."""

from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET


def prepare_recording(sumo_cmd, output_dir, seed, episode, distribution):
    """Enable outputs without changing vehicle behavior or signal timing."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = Path(sumo_cmd[sumo_cmd.index("-c") + 1]).resolve()
    config = ET.parse(config_path).getroot()
    net_path = (config_path.parent / config.find("input/net-file").get("value")).resolve()
    routes_path = (config_path.parent / config.find("input/route-files").get("value")).resolve()
    shutil.copyfile(net_path, output_dir / "environment.net.xml")
    shutil.copyfile(routes_path, output_dir / "episode_routes.rou.xml")

    originals = []
    additional = config.find("input/additional-files")
    if additional is not None:
        originals = [str((config_path.parent / p.strip()).resolve())
                     for p in additional.get("value").split(",")]
    # Keep the original additional files; a command-line override otherwise
    # drops tls.add.xml and silently changes the experiment.
    tls_path = config_path.parent / "tls.add.xml"
    shutil.copyfile(tls_path, output_dir / "tls.add.xml")
    recorder = ET.Element("additional")
    ET.SubElement(recorder, "timedEvent", type="SaveTLSStates", source="TL",
                  dest=str(output_dir / "tls_states.xml.gz"))
    recorder_path = output_dir / "recording.add.xml"
    ET.ElementTree(recorder).write(recorder_path, encoding="utf-8", xml_declaration=True)

    # Empty-network visual playback: the replay script supplies recorded
    # vehicle positions as POIs and restores the recorded signal states.
    replay = ET.Element("configuration")
    inputs = ET.SubElement(replay, "input")
    ET.SubElement(inputs, "net-file", value="environment.net.xml")
    ET.SubElement(inputs, "additional-files", value="tls.add.xml")
    ET.ElementTree(replay).write(output_dir / "replay.sumocfg", encoding="utf-8",
                                xml_declaration=True)
    command = list(sumo_cmd) + [
        "--additional-files", ",".join(originals + [str(recorder_path)]),
        "--fcd-output", str(output_dir / "fcd.xml.gz"),
        "--tripinfo-output", str(output_dir / "tripinfo.xml.gz"),
        "--tripinfo-output.write-unfinished", "true",
    ]
    with (output_dir / "recording.json").open("w", encoding="utf-8") as stream:
        json.dump({"traffic_seed": seed, "episode": episode, "distribution": distribution,
                   "files": {"trajectories": "fcd.xml.gz", "signals": "tls_states.xml.gz",
                             "tripinfo": "tripinfo.xml.gz", "routes": "episode_routes.rou.xml",
                             "replay_config": "replay.sumocfg"},
                   "recording_command": command,
                   "playback": "python Code/replay_episode.py --episode-dir THIS_DIRECTORY"},
                  stream, indent=2)
    return command
