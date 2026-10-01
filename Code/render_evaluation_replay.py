"""Render a saved SUMO evaluation as GIF/MP4, without rerunning its policy.

Vehicle positions/angles and traffic-light states come from the recorded XML.
Requires Matplotlib/Pillow; MP4 additionally requires ffmpeg on PATH.
"""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from replay_episode import xml_records


def shape_points(shape):
    return [tuple(map(float, point.split(",")[:2])) for point in shape.split()]


def load_frames(directory, start, end, step):
    frames = []
    for sample in xml_records(directory / "fcd.xml.gz", "timestep"):
        time = float(sample.get("time"))
        if time > end:
            break
        if time < start or (frames and time < frames[-1]["time"] + step - 1e-8):
            continue
        frames.append({"time": time, "vehicles": [dict(v.attrib) for v in sample.findall("vehicle")]})
    if not frames:
        raise ValueError("No recorded frames in the requested time window")
    signals = xml_records(directory / "tls_states.xml.gz", "tlsState")
    upcoming = next(signals, None)
    current = None
    try:
        for frame in frames:
            while upcoming is not None and float(upcoming.get("time")) <= frame["time"]:
                if upcoming.get("id") == "TL":
                    current = dict(upcoming.attrib)
                upcoming = next(signals, None)
            if current is None:
                raise ValueError(f"Missing recorded TL state at time {frame['time']}")
            frame["signal"] = current.copy()
    finally:
        signals.close()
    return frames


def vehicle_polygon(vehicle):
    # FCD position is the front bumper; angles are clockwise from North.
    angle = math.radians(float(vehicle.get("angle", 0)))
    dx, dy = math.sin(angle), math.cos(angle)
    rx, ry = dy, -dx
    x, y = float(vehicle["x"]), float(vehicle["y"])
    return [(x + sign * .9 * rx - back * dx, y + sign * .9 * ry - back * dy)
            for back, sign in ((0, 1), (5, 1), (5, -1), (0, -1))]


def create_animation(directory, frames, radius, fps, fig=None, axis=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    network = ET.parse(directory / "environment.net.xml").getroot()
    junction = next(j for j in network.findall("junction") if j.get("id") == "TL")
    cx, cy = float(junction.get("x")), float(junction.get("y"))
    lanes = {lane.get("id"): lane for edge in network.findall("edge") for lane in edge.findall("lane")}
    lane_shapes = {key: shape_points(lane.get("shape")) for key, lane in lanes.items()}
    incoming = {lane.get("id") for edge in network.findall("edge") if edge.get("to") == "TL"
                for lane in edge.findall("lane")}
    links = [connection for connection in network.findall("connection")
             if connection.get("tl") == "TL" and connection.get("via") in lane_shapes]

    if fig is None:
        fig, axis = plt.subplots(figsize=(8, 8), dpi=110)
        fig.patch.set_facecolor("#edf2f4")
        fig.subplots_adjust(left=.06, right=.94, bottom=.13, top=.85)
    axis.set_facecolor("#edf2f4")
    axis.set_aspect("equal")
    axis.set_xlim(cx-radius, cx+radius)
    axis.set_ylim(cy-radius, cy+radius)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    # Convert lane widths in meters into line widths in points.
    box = axis.get_position()
    lane_scale = min(fig.get_figwidth()*box.width, fig.get_figheight()*box.height) * 72 / (2 * radius)
    axis.add_collection(LineCollection(list(lane_shapes.values()), colors="#34424f",
                        linewidths=[float(l.get("width", 3.2)) * lane_scale for l in lanes.values()],
                        zorder=1, capstyle="butt"))
    if junction.get("shape"):
        axis.add_patch(Polygon(shape_points(junction.get("shape")), facecolor="#34424f",
                               edgecolor="none", zorder=1))
    axis.add_collection(LineCollection(list(lane_shapes.values()), colors="#73828c",
                                       linewidths=.45, linestyles="dashed", zorder=2))
    movement_paths = LineCollection([lane_shapes[c.get("via")] for c in links],
                                    linewidths=1.6, alpha=.8, zorder=3)
    axis.add_collection(movement_paths)
    cars = PolyCollection([], edgecolors="#17212b", linewidths=.35, zorder=4)
    axis.add_collection(cars)
    status = axis.text(.025, .97, "", transform=axis.transAxes, va="top", fontsize=10,
                       bbox={"facecolor": "white", "alpha": .94, "edgecolor": "none", "pad": 6},
                       zorder=10)
    for label, x, y in (("N", cx, cy+radius*.87), ("S", cx, cy-radius*.87),
                        ("E", cx+radius*.87, cy), ("W", cx-radius*.87, cy)):
        axis.text(x, y, label, ha="center", va="center", fontsize=12, weight="bold", color="#17212b")
    axis.plot([cx-radius*.9, cx-radius*.9+25], [cy-radius*.9]*2, color="#17212b", lw=2)
    axis.text(cx-radius*.9+12.5, cy-radius*.9+4, "25 m", ha="center", fontsize=9)

    run_config = directory.parents[2] / "config.json"
    config = json.loads(run_config.read_text()) if run_config.exists() else {}
    recording = json.loads((directory / "recording.json").read_text())
    volume = directory.parent.name.removeprefix("volume_")
    title = config.get("controller", "Saved evaluation")
    fig.suptitle(f"{title}  |  demand {volume} vehicles\nTraffic seed {recording['traffic_seed']}  |  recorded evaluation replay",
                 fontsize=14, color="#17212b", y=.96)
    legends = [Line2D([], [], marker="s", color="none", markerfacecolor=color, markersize=7, label=label)
               for label, color in (("Moving vehicle", "#369ce0"), ("Stopped vehicle", "#ffae42"),
                                    ("Green movement", "#27c779"), ("Yellow", "#ffc647"),
                                    ("Red movement", "#ef6262"))]
    fig.legend(handles=legends, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, .045))
    fig.text(.5, .025, f"Recorded positions and signal states - not a new simulation | {fps * (frames[1]['time']-frames[0]['time']) if len(frames)>1 else 1:g}x playback",
             ha="center", fontsize=9, color="#475761")
    colors = {"g": "#27c779", "y": "#ffc647", "r": "#ef6262"}

    def update(index):
        frame = frames[index]
        visible = [v for v in frame["vehicles"] if abs(float(v["x"])-cx) <= radius
                   and abs(float(v["y"])-cy) <= radius]
        cars.set_verts([vehicle_polygon(v) for v in visible])
        cars.set_facecolors(["#ffae42" if float(v.get("speed", 0)) < .1 else "#369ce0" for v in visible])
        state = frame["signal"]["state"]
        movement_paths.set_colors([colors.get(state[int(c.get("linkIndex"))].lower(), "#8b99a5") for c in links])
        stopped = sum(v.get("lane") in incoming and float(v.get("speed", 0)) < .1 for v in visible)
        status.set_text(f"Simulation time: {frame['time']:.0f} s\nSUMO phase: {frame['signal']['phase']}\nVehicles in view: {len(visible)}\nStopped incoming in view: {stopped}")
        return cars, movement_paths, status

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    return fig, animation, update


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .gif or .mp4 file")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=120)
    parser.add_argument("--step", type=float, default=1, help="Simulation seconds between rendered frames")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--radius", type=float, default=130, help="Intersection crop radius, meters")
    parser.add_argument("--presentation", action="store_true", help="1080p widescreen with demand and queue charts")
    args = parser.parse_args(argv)
    if args.start < 0 or args.end < args.start or args.step <= 0 or args.fps <= 0 or args.radius <= 0:
        parser.error("Invalid time window, step, fps, or radius")
    if args.output.suffix.lower() not in (".gif", ".mp4"):
        parser.error("--output must end in .gif or .mp4")
    outputs = [args.output, args.output.with_suffix(".png"), args.output.with_suffix(".json")]
    if args.presentation:
        outputs.append(args.output.with_suffix('.metrics.csv'))
    if any(p.exists() for p in outputs):
        parser.error("Output or sidecar file already exists; choose a new filename")
    from matplotlib.animation import FFMpegWriter, PillowWriter, writers
    if args.output.suffix.lower() == ".mp4" and not writers.is_available("ffmpeg"):
        parser.error("MP4 export requires ffmpeg; use .gif instead")
    frames = load_frames(args.episode_dir, args.start, args.end, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, animation, update = create_animation(args.episode_dir, frames, args.radius, args.fps)
    presentation_metadata = {}
    if args.presentation:
        from presentation_replay import enhance_presentation
        animation, update, presentation_metadata = enhance_presentation(
            fig, animation, update, args.episode_dir, frames, args.fps, args.output)
    print(f"Rendering {len(frames)} frames to {args.output}", flush=True)
    writer = (PillowWriter(fps=args.fps) if args.output.suffix.lower() == ".gif"
              else FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
    dpi = 120 if args.presentation else 110
    animation.save(str(args.output), writer=writer, dpi=dpi)
    update(len(frames)//2)
    fig.savefig(outputs[1], dpi=dpi)
    metadata = {"source_episode": str(args.episode_dir.resolve()), "type": "recorded evaluation replay",
                "simulation_start": frames[0]["time"], "simulation_end": frames[-1]["time"],
                "rendered_frames": len(frames), "fps": args.fps, "crop_radius_m": args.radius,
                "phase_indices": sorted({int(f["signal"]["phase"]) for f in frames}),
                "note": "Stopped incoming counts are reconstructed from rounded FCD speeds within the crop, not original full-episode metrics."}
    metadata.update(presentation_metadata)
    if args.presentation:
        metadata['note'] = 'Reconstructed replay metrics; see definitions. Single episode, not aggregate evidence.'
    outputs[2].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Saved animation and preview: {args.output}, {outputs[1]}")


if __name__ == "__main__":
    main()
