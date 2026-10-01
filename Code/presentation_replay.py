"""Data-derived presentation panels for recorded SUMO episodes."""

from bisect import bisect_right
import csv
import json
import xml.etree.ElementTree as ET

from replay_episode import xml_records


def collect_metrics(directory):
    """Read every FCD sample, independently of animation frame sampling."""
    network = ET.parse(directory / 'environment.net.xml').getroot()
    junctions = {j.get('id'): j for j in network.findall('junction')}
    center = junctions['TL']
    incoming = {}
    for edge in network.findall('edge'):
        if edge.get('to') != 'TL':
            continue
        origin = junctions[edge.get('from')]
        dx = float(origin.get('x')) - float(center.get('x'))
        dy = float(origin.get('y')) - float(center.get('y'))
        direction = ('E' if dx > 0 else 'W') if abs(dx) > abs(dy) else ('N' if dy > 0 else 'S')
        incoming.update({lane.get('id'): direction for lane in edge.findall('lane')})
    scheduled = sorted(float(v.get('depart')) for v in
                       ET.parse(directory / 'episode_routes.rou.xml').getroot().findall('vehicle'))
    trips = [dict(t.attrib) for t in xml_records(directory / 'tripinfo.xml.gz', 'tripinfo')]
    departed = sorted(float(t['depart']) for t in trips if float(t['depart']) >= 0)
    completed = sorted(float(t['arrival']) for t in trips
                       if float(t['arrival']) >= 0 and t.get('vaporized', '') in ('', '0', 'false'))
    rows = []
    for sample in xml_records(directory / 'fcd.xml.gz', 'timestep'):
        time = float(sample.get('time'))
        vehicles = sample.findall('vehicle')
        queues = dict.fromkeys('NESW', 0)
        for v in vehicles:
            direction = incoming.get(v.get('lane'))
            if direction and float(v.get('speed')) < .1:
                queues[direction] += 1
        rows.append(dict(time_s=time, active=len(vehicles),
                         scheduled=bisect_right(scheduled, time),
                         inserted=bisect_right(departed, time),
                         completed=bisect_right(completed, time),
                         inserted_last_60s=bisect_right(departed, time)-bisect_right(departed, time-60),
                         stopped_incoming=sum(queues.values()),
                         **{f'stopped_{d}': queues[d] for d in 'NESW'}))
    return rows, scheduled


def enhance_presentation(fig, animation, map_update, directory, frames, fps, output):
    import numpy as np
    from matplotlib.lines import Line2D

    rows, scheduled = collect_metrics(directory)
    times = [r['time_s'] for r in rows]
    minutes = np.array(times) / 60
    end = (times[-1] + (times[1]-times[0] if len(times)>1 else 1)) / 60
    config_path = directory.parents[2] / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    recording = json.loads((directory / 'recording.json').read_text())
    fig.set_size_inches(16, 9)
    fig.set_dpi(120)
    fig.patch.set_facecolor('#f5f7fa')
    axis = fig.axes[0]
    axis.set_position([.025, .235, .49, .565])
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()
    map_update(0)[2].set_visible(False)
    fig.text(.035, .945, 'Adaptive traffic signal control', size=25, weight='bold', color='#152b40')
    speed = fps * (frames[1]['time']-frames[0]['time']) if len(frames)>1 else 1
    fig.text(.035, .902,
             f"{config.get('controller', 'Saved controller')}  |  {len(scheduled):,} scheduled vehicles  |  "
             f"{recording.get('distribution', 'Recorded')} demand  |  seed {recording['traffic_seed']}  |  {speed:g}× playback",
             size=13, color='#42566a')
    clock = fig.text(.965, .944, '', ha='right', size=22, weight='bold', color='#152b40')
    cards = []
    for x, label in ((.035, 'INSERTED'), (.165, 'TRIPS COMPLETED'), (.325, 'ACTIVE IN NETWORK'), (.535, 'STOPPED ON INCOMING ROADS')):
        fig.text(x, .85, label, size=10, weight='bold', color='#506277')
        cards.append(fig.text(x, .803, '', size=24, weight='bold', color='#152b40'))
    phase = fig.text(.035, .212, '', size=12, color='#152b40')
    approach = fig.text(.035, .174, '', size=12, color='#152b40')
    fig.legend(handles=[Line2D([], [], marker='s', linestyle='', color=c, label=l)
                        for c, l in (('#369ce0', 'Moving'), ('#ffae42', 'Stopped'),
                                     ('#27c779', 'Green movement'), ('#ffc647', 'Yellow'),
                                     ('#ef6262', 'Red'))],
               loc='center', bbox_to_anchor=(.265, .125), ncol=3, frameon=False, fontsize=10)

    def panel(bounds, title, ylabel, ymax):
        ax = fig.add_axes(bounds, facecolor='white')
        ax.set_title(title, loc='left', size=13, weight='bold', pad=12, color='#152b40')
        ax.set_xlim(0, end)
        ax.set_ylim(0, max(1, ymax))
        ax.set_ylabel(ylabel, size=10)
        ax.tick_params(labelsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#c4cdd5')
        ax.grid(axis='y', color='#e5e9ed', linewidth=.7)
        return ax

    bins = np.arange(0, max(60, times[-1]+60)+.001, 60)
    histogram, edges = np.histogram(scheduled, bins=bins)
    flow = panel([.565, .525, .40, .205], 'Demand over time', 'Vehicles / minute',
                 max(max(histogram), max(r['inserted_last_60s'] for r in rows))*1.3)
    flow.stairs(histogram, edges/60, color='#8999a8', fill=True, alpha=.20,
                label='Scheduled (fixed 1-min bins)')
    flow_line, = flow.plot([], [], color='#0072b2', lw=2, label='Inserted (previous 60 s)')
    flow.legend(loc='upper right', fontsize=9, frameon=False)
    flow_cursor = flow.axvline(0, color='#172b40', lw=1, ls='--')
    queue = panel([.565, .205, .40, .205], 'Queue response · all incoming roads', 'Stopped vehicles',
                  max(r['stopped_incoming'] for r in rows)*1.2)
    queue_line, = queue.plot([], [], color='#d55e00', lw=2)
    queue_cursor = queue.axvline(0, color='#172b40', lw=1, ls='--')
    queue.set_xlabel('Simulation time (minutes)', size=11)
    queue_note = fig.text(.565, .132, '', size=11, color='#42566a')
    fig.text(.035, .063, 'Recorded SUMO evaluation  •  Map: intersection close-up  •  Counts and charts: full network / incoming roads',
             size=10, color='#506277')
    fig.text(.035, .034, 'Stopped: recorded speed < 0.1 m/s. Charts use every recorded sample. Single episode; comparative performance requires repeated evaluations.',
             size=9, color='#506277')
    progress = fig.add_axes([.035, .012, .93, .006])
    progress.set_xlim(0, end*60)
    progress.set_ylim(0, 1)
    progress.axis('off')
    progress.axhspan(0, 1, color='#dfe5eb')
    bar = progress.barh(.5, 0, height=1, color='#0072b2')[0]
    cumulative_peak = np.maximum.accumulate([r['stopped_incoming'] for r in rows])
    flow_values = [r['inserted_last_60s'] for r in rows]
    queue_values = [r['stopped_incoming'] for r in rows]

    def update(index):
        artists = map_update(index)
        t = frames[index]['time']
        i = bisect_right(times, t)-1
        row = rows[i]
        clock.set_text(f'{int(t)//60:02d}:{int(t)%60:02d} / {int(end):02d}:00')
        for card, key in zip(cards, ('inserted', 'completed', 'active', 'stopped_incoming')):
            card.set_text(f'{row[key]:,}')
        state = frames[index]['signal']
        phase.set_text(f"Recorded SUMO phase {state['phase']}  |  G: {sum(c in 'Gg' for c in state['state'])}  "
                       f"Y: {sum(c in 'Yy' for c in state['state'])}  R: {sum(c in 'Rr' for c in state['state'])} controlled links")
        approach.set_text('Stopped by approach    ' + '    '.join(f"{d}: {row['stopped_'+d]}" for d in 'NESW'))
        flow_line.set_data(minutes[:i+1], flow_values[:i+1])
        queue_line.set_data(minutes[:i+1], queue_values[:i+1])
        flow_cursor.set_xdata([t/60, t/60])
        queue_cursor.set_xdata([t/60, t/60])
        queue_note.set_text(f"Peak stopped so far: {cumulative_peak[i]} vehicles  |  Incoming roads: N + E + S + W")
        bar.set_width(t)
        return artists

    # FuncAnimation calls this public callback through a new animation instance.
    from matplotlib.animation import FuncAnimation
    animation._draw_was_started = True  # Suppress warning for the replaced map-only animation.
    new_animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False)
    csv_path = output.with_suffix('.metrics.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {'layout': 'presentation', 'metrics_csv': csv_path.name,
                  'metric_samples': len(rows), 'scheduled_vehicles': len(scheduled),
                  'stopped_definition': 'FCD rounded speed < 0.1 m/s on all lanes entering TL',
                  'inserted_definition': 'TripInfo actual depart >= 0 and <= displayed time',
                  'completed_definition': 'TripInfo arrival >= 0 and <= displayed time; excludes vaporized trips',
                  'flow_definition': 'Inserted in (time - 60, time]; scheduled in fixed 60-second bins',
                  'sources': ['https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html',
                              'https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html',
                              'https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/']}
    return new_animation, update, provenance
