"""Visual playback of recorded vehicle positions and actual signal states.

Vehicles are moving POIs, not a newly simulated policy. PyTorch is not needed.
"""

import argparse
import gzip
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def xml_records(path, tag):
    """Stream compressed XML without loading a whole episode into RAM."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        context = ET.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        for event, element in context:
            if event == "end" and element.tag == tag:
                yield element
                root.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", help="Validate playback without a GUI")
    parser.add_argument("--delay", type=int, default=30, help="GUI delay in milliseconds")
    parser.add_argument("--end-time", type=float, help="Stop playback at this recorded time")
    args = parser.parse_args(argv)
    if "SUMO_HOME" in os.environ:
        sys.path.append(str(Path(os.environ["SUMO_HOME"]) / "tools"))
    import traci
    from sumolib import checkBinary

    directory = args.episode_dir.resolve()
    for name in ("replay.sumocfg", "fcd.xml.gz", "tls_states.xml.gz"):
        if not (directory / name).is_file():
            parser.error(f"Missing recording: {directory / name}")
    command = [checkBinary("sumo" if args.headless else "sumo-gui"),
               "-c", str(directory / "replay.sumocfg"), "--no-step-log", "true"]
    if not args.headless:
        command += ["--delay", str(args.delay), "--start"]
    signals = xml_records(directory / "tls_states.xml.gz", "tlsState")
    signal = next(signals, None)
    active = set()
    frames = 0
    traci.start(command)
    try:
        for frame in xml_records(directory / "fcd.xml.gz", "timestep"):
            time = float(frame.get("time"))
            if args.end_time is not None and time > args.end_time:
                break
            if time > traci.simulation.getTime():
                traci.simulationStep(time)
            while signal is not None and float(signal.get("time")) <= time:
                traci.trafficlight.setRedYellowGreenState(signal.get("id"), signal.get("state"))
                signal = next(signals, None)
            current = set()
            for vehicle in frame.findall("vehicle"):
                vehicle_id = vehicle.get("id")
                x, y = float(vehicle.get("x")), float(vehicle.get("y"))
                if vehicle_id in active:
                    traci.poi.setPosition(vehicle_id, x, y)
                else:
                    traci.poi.add(vehicle_id, x, y, (40, 130, 220, 255), width=4, height=4)
                traci.poi.setParameter(vehicle_id, "recorded_speed", vehicle.get("speed", ""))
                current.add(vehicle_id)
            for vehicle_id in active - current:
                traci.poi.remove(vehicle_id)
            active = current
            frames += 1
        print(f"Replayed {frames} recorded timesteps")
    finally:
        traci.close()
        signals.close()


if __name__ == "__main__":
    main()
