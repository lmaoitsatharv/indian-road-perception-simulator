"""Publish the verified 20 October 2026 observation report and portable summary."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--portable",type=Path)
    args=parser.parse_args()
    report=json.loads((args.output/'report.json').read_text(encoding='utf-8'))
    independent=json.loads((args.output/'independent_verification.json').read_text(encoding='utf-8'))
    assert report['status']=='completed' and independent['passed']
    assert args.output.name == 'carla_road_observation_20261020_01'
    assert [e['seed'] for e in report['episodes']] == [7,42]
    assert all(e['frames'] == 550 and e['render_quality'] == 'Low'
        and e['native_run']['ego_goal_reached'] and e['native_run']['completed_overtakes'] == 1
        and e['native_run']['collision_events'] == e['native_run']['sampled_body_overlaps'] == 0
        for e in report['episodes'])
    def percent(x):
        return f'{100*x:.2f}%' if x is not None else 'n/a'
    summary={k:report[k] for k in ('kind','created_utc','metrics','model_control','model_weights_unchanged',
        'protected_files_unchanged','reference_mapping','resize','source_to_model_pixel_centers','limits','warmup_ms','runtime')}
    summary['independent_verification']=independent
    summary['player_verification']={'embedded_report_matches':True,'comparison_images_present':1100,
        'javascript_syntax_checked':True,'playback_controls_automated':False,
        'limitation':'Browser URL policy blocked local-file observation; no workaround was used'}
    summary['result_directory']='runs/'+args.output.name
    summary['episodes']=[]
    lines=['# Simulator-camera road-model observation — 20 October 2026 IST','',
        '[Open the comparison replay](index.html).','',
        'The locked morning-best model was applied to recorded front-camera frames from two selected mixed-traffic seeds. '
        'Capture and inference ran separately. The rule driver used simulator state throughout; predictions did not control the car.','',
        '| Seed | Frames | Road IoU | Dice | Precision | Recall | Evaluated coverage |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for episode in report['episodes']:
        m=episode['metrics']
        lines.append('| '+str(episode['seed'])+' | '+str(episode['frames'])+' | '+' | '.join(percent(m[k]) for k in
            ('road_iou','dice','precision','recall','evaluated_fraction'))+' |')
        item={k:episode[k] for k in ('seed','case','frames','render_quality','metrics','class_breakdown',
            'inference_ms','processing_wall_s_including_export','native_run','source_manifest_sha256')}
        item['worst_frames']=[{'frame':r['frame'],'sim_s':r['sim_s'],'road_iou':r['metrics']['road_iou'],'comparison':r['comparison']}
            for r in sorted(episode['rows'],key=lambda r:r['metrics']['road_iou'] if r['metrics']['road_iou'] is not None else 2)[:8]]
        summary['episodes'].append(item)
    m=report['metrics']
    lines+=['',f"Pooled road IoU: **{percent(m['road_iou'])}**. Coverage: **{percent(m['evaluated_fraction'])}**. "
        'Adjacent frames and these related seeds are correlated development observations, not independent test samples.','',
        '## What the reference measures','',
        'Native road/sidewalk tags were absent on the procedural map. Raw tags are retained. The reference uses aligned depth '
        'and the saved OpenDRIVE driving-lane footprint to resolve visible unlabeled surfaces; native actors remain non-road. '
        'Uncertain edges and unknown distant geometry are excluded. Purple pixels are excluded, not model road labels. '
        'Parking bays and shared paved areas are outside the driving-lane definition, so some disagreement reflects the definition.','',
        'This score is agreement with a geometry-assisted simulator reference. It is separate from the completed real-image test '
        'and is not a new estimate of real-road accuracy. The reference raster and junction footprint have finite precision.','',
        '## Measured failures and timings','']
    for episode in report['episodes']:
        worst=min(episode['rows'],key=lambda r:r['metrics']['road_iou'] if r['metrics']['road_iou'] is not None else 2)
        timing=episode['inference_ms']
        lines.append(f"- Seed {episode['seed']}: worst frame at {worst['sim_s']:.2f} s, road IoU {percent(worst['metrics']['road_iou'])}. "
            f"[View comparison]({worst['comparison']}). Model computation median {timing['median']:.2f} ms, p95 {timing['p95']:.2f} ms. "
            f"Render quality: {episode['render_quality']}.")
    lines+=['','The broad, visually plain junction produces large missed-road patches. Some road/pavement boundaries '
        'and paved margins produce extra-road predictions. Per-class pixel fractions are saved in report.json and the player; '
        'they are not actor-detection recall. Inspect those failures before considering camera-based control.','',
        'Timings above measure the model call only, excluding capture, transfer, geometry processing, resize and report export. '
        'The saved camera cadence is 5 Hz; neither this replay nor the isolated model timing establishes live driving-loop latency.','',
        '## Verification','',
        f"- Recounted {independent['frames_recounted']} saved binary prediction/reference pairs independently.",
        f"- Queried {independent['projected_road_points']} depth-projected road points against OpenDRIVE; "
        f"{independent['points_outside_opendrive_driving_lanes']} were outside driving lanes.",
        f"- Checked {independent['known_actor_pixels_checked']} known-actor pixels for reference occlusion integrity.",
        '- Retained probability examples match the fixed >=0.5 binary threshold.',
        '- The embedded player data, all 1,100 comparison-image paths and JavaScript syntax passed static checks. '
        'Browser automation blocked local-file pages, so playback controls were not automatically exercised.',
        '- All three checkpoints, lock/selection metadata, split manifest and completed test reports retained their fingerprints. '
        'Loaded model weights also retained their fingerprint after inference.',
        '- Both native captures reached their goal, completed one overtake, recorded zero native collisions and sampled overlaps, '
        'and closed their owned jobs.','',
        'The seed-7 camera run retained the prior goal time and categorical decisions, but its native actor trace was not bitwise '
        'identical to the earlier Epic-render baseline: maximum position difference 0.03473 m, target-speed difference '
        '0.30121 m/s, and throttle difference 0.17949. The policy source remained unchanged. These observed differences '
        'must not be presented as exact replay equivalence.','',
        '## Preserved failures','',
        '- The first semantic-only probe showed no road-tag pixels; it was not scored as a valid road reference.',
        '- One full Epic-render retry crashed natively during traffic setup with EXCEPTION_ACCESS_VIOLATION before observation '
        'cameras started. Its evidence remains in runs/carla_india_smoke_20261019T234718_109014Z_seed7. '
        'The scored full captures used Low rendering quality and clean owned-job shutdown.','',
        '## Next stage','',
        'Review the failures and improve scene/reference realism, then add bounded live observation with explicit frame age '
        'and an independent vehicle/pedestrian detection/tracking path. Future RL stays STOP / SLOW / PROCEED above the '
        'route controller and independent safety checks. No RL training or segmentation tuning occurred here.','']
    (args.output/'OBSERVATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if args.portable:
        args.portable.parent.mkdir(parents=True,exist_ok=True)
        args.portable.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(args.output/'OBSERVATION_REPORT.md'),'portable':str(args.portable),
                      'frames':independent['frames_recounted'],'road_iou':m['road_iou']}))


if __name__=='__main__':
    main()
