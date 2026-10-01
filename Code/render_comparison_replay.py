"""Synchronized LSTM/GRU comparison with a reproducible representative seed.

Default: select one matched episode near each model's median mean stopped
incoming count across all seeds, then render the full hour in 60 seconds.
"""

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
from pathlib import Path
import statistics

from presentation_replay import collect_metrics
from render_evaluation_replay import create_animation, load_frames


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_pair(first_root, second_root, volume, seed=None):
    first_volume = first_root / 'replays' / f'volume_{volume}'
    second_volume = second_root / 'replays' / f'volume_{volume}'
    names = sorted(p.name for p in first_volume.iterdir() if p.is_dir())
    other = sorted(p.name for p in second_volume.iterdir() if p.is_dir())
    if names != other or not names:
        raise ValueError('Both models must have the same nonempty set of episode directories')
    summaries = []
    datasets = {}
    for name in names:
        first, second = first_volume / name, second_volume / name
        if digest(first / 'episode_routes.rou.xml') != digest(second / 'episode_routes.rou.xml'):
            raise ValueError(f'Route files differ for {name}')
        if digest(first / 'environment.net.xml') != digest(second / 'environment.net.xml'):
            raise ValueError(f'Network files differ for {name}')
        a, scheduled_a = collect_metrics(first)
        b, scheduled_b = collect_metrics(second)
        if [r['time_s'] for r in a] != [r['time_s'] for r in b]:
            raise ValueError(f'Recorded time grids differ for {name}')
        if scheduled_a != scheduled_b or len(scheduled_a) != volume:
            raise ValueError(f'Scheduled demand does not match volume for {name}')
        summary = {'seed': int(name.removeprefix('seed_'))}
        for label, rows in (('lstm', a), ('gru', b)):
            summary[f'{label}_mean_stopped'] = statistics.mean(r['stopped_incoming'] for r in rows)
            summary[f'{label}_peak_stopped'] = max(r['stopped_incoming'] for r in rows)
            summary[f'{label}_completed_final'] = rows[-1]['completed']
        summaries.append(summary)
        datasets[name] = (a, b, scheduled_a)
        print(f"Analyzed volume {volume}, {name}", flush=True)
    medians = {label: statistics.median(r[f'{label}_mean_stopped'] for r in summaries)
               for label in ('lstm', 'gru')}
    # Standard deviation normalizes the distance; ties resolve to the lowest seed.
    scales = {label: statistics.pstdev(r[f'{label}_mean_stopped'] for r in summaries) or 1
              for label in ('lstm', 'gru')}
    for row in summaries:
        row['selection_distance'] = sum(
            ((row[f'{label}_mean_stopped']-medians[label])/scales[label])**2
            for label in ('lstm', 'gru'))
    candidates = [r for r in summaries if seed is None or r['seed'] == seed]
    if not candidates:
        raise ValueError(f'Seed {seed} not found')
    chosen = min(candidates, key=lambda r: (r['selection_distance'], r['seed']))
    name = f"seed_{chosen['seed']}"
    return first_volume/name, second_volume/name, datasets[name], {
        'selected_seed': chosen['seed'], 'episodes_analyzed_per_model': len(names),
        'rule': 'Minimum summed squared standardized distance from each model median mean stopped incoming count; lowest seed breaks ties',
        'override_seed': seed, 'medians': medians, 'normalization_sd': scales,
        'all_pairs': summaries}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison(first, second, datasets, frames_a, frames_b, radius, fps, volume, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    import numpy as np

    a, b, scheduled = datasets
    times = [r['time_s'] for r in a]
    minutes = np.array(times)/60
    end = (times[-1]+(times[1]-times[0] if len(times)>1 else 1))/60
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor='#f5f7fa')
    maps = [fig.add_axes([x, .315, .44, .485]) for x in (.03, .53)]
    updates = []
    for directory, frames, axis in zip((first, second), (frames_a, frames_b), maps):
        _, animation, update = create_animation(directory, frames, radius, fps, fig=fig, axis=axis)
        animation._draw_was_started = True
        update(0)[2].set_visible(False)
        updates.append(update)
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()
    colors = ('#0072b2', '#cc79a7')
    labels = ('PPO-LSTM', 'PPO-GRU')
    speed = fps*(frames_a[1]['time']-frames_a[0]['time']) if len(frames_a)>1 else 1
    fig.text(.035, .951, 'Variable demand | synchronized controller comparison',
             size=23, weight='bold', color='#152b40')
    fig.text(.035, .912, f'{volume:,} scheduled vehicles | identical routes | matched seed {seed} | {speed:g}x playback',
             size=13, color='#42566a')
    clock = fig.text(.965, .912, '', ha='right', size=15, weight='bold', color='#152b40')
    stat_texts, phase_texts, approach_texts = [], [], []
    for x, label, color in zip((.035, .535), labels, colors):
        fig.text(x, .862, label, size=20, weight='bold', color=color)
        stat_texts.append(fig.text(x, .819, '', size=12, color='#152b40'))
        phase_texts.append(fig.text(x, .295, '', size=11, color='#152b40'))
        approach_texts.append(fig.text(x, .267, '', size=11, color='#152b40'))
    fig.legend(handles=[Line2D([], [], marker='s', linestyle='', color=c, label=l)
                        for c, l in (('#369ce0', 'Moving vehicle'), ('#ffae42', 'Stopped vehicle'),
                                     ('#27c779', 'Green movement'), ('#ffc647', 'Yellow'), ('#ef6262', 'Red'))],
               loc='center', bbox_to_anchor=(.5, .239), ncol=5, frameon=False, fontsize=9)
    axes, cursors = [], []
    for x, title, ylabel, ymax in (
            (.07, 'Scheduled demand', 'Vehicles / min', 1),
            (.395, 'Stopped on incoming roads', 'Vehicles', max(r['stopped_incoming'] for r in a+b)*1.15),
            (.72, 'Trips completed', 'Vehicles', volume*1.05)):
        axis = fig.add_axes([x, .076, .265, .125], facecolor='white')
        axis.set_title(title, loc='left', fontsize=11, weight='bold', color='#152b40', pad=8)
        axis.set_xlim(0, end)
        axis.set_ylim(0, max(1, ymax))
        axis.set_ylabel(ylabel, size=9)
        axis.set_xlabel('Simulation time (min)', size=9, labelpad=2)
        axis.tick_params(labelsize=9)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color='#e5e9ed', linewidth=.7)
        axes.append(axis)
        cursors.append(axis.axvline(0, color='#42566a', lw=1, ls='--'))
    bins = np.arange(0, end*60+60, 60)
    counts, edges = np.histogram(scheduled, bins=bins)
    axes[0].stairs(counts, edges/60, fill=True, color='#8497a8', alpha=.5)
    axes[0].set_ylim(0, max(1, max(counts)*1.15))
    lines = []
    for axis, key in zip(axes[1:], ('stopped_incoming', 'completed')):
        pair = [axis.plot([], [], color=color, lw=1.5, label=label)[0]
                for label, color in zip(labels, colors)]
        axis.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
        lines.append((pair, key))
    fig.text(.035, .027, 'Recorded SUMO positions and signals | Map: close-up; counts: full network / incoming roads | One illustrative matched episode',
             size=9, color='#506277')
    fig.text(.035, .010, 'Stopped = FCD speed < 0.1 m/s on lanes entering TL. Charts use every recorded second; maps sample every 3 s at default settings.',
             size=8, color='#506277')

    def update(index):
        t = frames_a[index]['time']
        i = bisect_right(times, t)-1
        clock.set_text(f'{int(t)//60:02d}:{int(t)%60:02d} / {int(end):02d}:00')
        for j, (rows, frames, map_update) in enumerate(zip((a, b), (frames_a, frames_b), updates)):
            map_update(index)
            row, signal = rows[i], frames[index]['signal']
            stat_texts[j].set_text(f"Inserted {row['inserted']:,}  |  Completed {row['completed']:,}  |  Active {row['active']:,}  |  Stopped {row['stopped_incoming']}")
            phase_texts[j].set_text(f"Recorded SUMO phase {signal['phase']} | Green {sum(c in 'gG' for c in signal['state'])}, yellow {sum(c in 'yY' for c in signal['state'])}, red {sum(c in 'rR' for c in signal['state'])} links")
            approach_texts[j].set_text('Stopped by approach:  ' + '   '.join(f"{d} {row['stopped_'+d]}" for d in 'NESW'))
        for pair, key in lines:
            for line, rows in zip(pair, (a, b)):
                line.set_data(minutes[:i+1], [r[key] for r in rows[:i+1]])
        for cursor in cursors:
            cursor.set_xdata([t/60, t/60])

    return fig, FuncAnimation(fig, update, frames=len(frames_a), interval=1000/fps), update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-root', type=Path, default=Path('Code/optimized_runs/LSTM_RUN'))
    parser.add_argument('--second-root', type=Path, default=Path('Code/optimized_runs/GRU_RUN'))
    parser.add_argument('--volume', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, help='Override representative-seed selection')
    parser.add_argument('--step', type=float, default=3)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--radius', type=float, default=130)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=3600)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.mp4' or min(args.step, args.fps, args.radius) <= 0 or args.start < 0 or args.end < args.start:
        parser.error('Use .mp4 output and valid time, sampling, and crop values')
    outputs = [args.output, args.output.with_suffix('.png'), args.output.with_suffix('.json'),
               args.output.with_suffix('.selection.csv'), args.output.with_suffix('.metrics.csv')]
    if any(p.exists() for p in outputs):
        parser.error('Output or sidecar already exists; choose another filename')
    from matplotlib.animation import FFMpegWriter, writers
    if not writers.is_available('ffmpeg'):
        parser.error('ffmpeg is required')
    first, second, datasets, selection = select_pair(args.first_root, args.second_root, args.volume, args.seed)
    print(f"Selected matched seed {selection['selected_seed']}", flush=True)
    frames_a = load_frames(first, args.start, args.end, args.step)
    frames_b = load_frames(second, args.start, args.end, args.step)
    if [f['time'] for f in frames_a] != [f['time'] for f in frames_b]:
        raise ValueError('Sampled frame times differ')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(outputs[3], selection['all_pairs'])
    metrics = [dict(controller=label, **r) for label, rows in zip(('PPO-LSTM', 'PPO-GRU'), datasets[:2]) for r in rows]
    write_csv(outputs[4], metrics)
    fig, animation, update = build_comparison(first, second, datasets, frames_a, frames_b,
                                             args.radius, args.fps, args.volume, selection['selected_seed'])
    update(0)
    fig.savefig(outputs[1], dpi=120)
    metadata = {'volume': args.volume, 'selection': selection,
                'source_episodes': [str(first.resolve()), str(second.resolve())],
                'simulation_start': frames_a[0]['time'], 'simulation_end': frames_a[-1]['time'],
                'frames': len(frames_a), 'fps': args.fps, 'step_s': args.step,
                'crop_radius_m': args.radius, 'metric_scope': 'Full network and all incoming lanes, independently of crop',
                'stopped_definition': 'Rounded recorded FCD speed < 0.1 m/s on lanes entering TL',
                'note': 'Post-hoc illustrative seed selection; not independent evidence of aggregate model superiority'}
    outputs[2].write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Rendering {len(frames_a)} synchronized frames to {args.output}', flush=True)
    animation.save(str(args.output), writer=FFMpegWriter(fps=args.fps, codec='libx264',
                   extra_args=['-pix_fmt', 'yuv420p']), dpi=120)
    # Choose a readable preview during the busy portion of the episode.
    preview = min(range(len(frames_a)), key=lambda i: abs(frames_a[i]['time']-600))
    update(preview)
    fig.savefig(outputs[1], dpi=120)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
