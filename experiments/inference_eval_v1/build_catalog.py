"""Resolve the frozen development cohorts against local data and model paths."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, required=True)
    parser.add_argument('--precompute-cache', type=Path, required=True)
    parser.add_argument('--empirical', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-id', default='aug2026')
    parser.add_argument('--base-catalog', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protected_directories = [args.datasets_root.resolve(), args.precompute_cache.resolve(), args.checkpoint.resolve().parent]
    protected_files = [args.empirical.resolve(), Path(__file__).with_name('cohorts.json').resolve()]
    if any(output.is_relative_to(root) for root in protected_directories) or output in protected_files:
        raise ValueError('Catalog output must be separate from recordings, preprocessing, model inputs, and the frozen roster')
    manifest = json.loads(Path(__file__).with_name('cohorts.json').read_text())
    catalog = json.loads(args.base_catalog.read_text()) if args.base_catalog else {'datasets': {}, 'models': {}}
    for record in manifest['records']:
        prefix = args.datasets_root.resolve() / record['basename']
        cache = args.precompute_cache.resolve()
        stem = prefix.stem + '_segmentation_nthetas65'
        for path in [prefix / 'data.mdb', prefix.with_suffix('.yaml'), cache / (stem + '.pkl'), cache / (stem + '.yarr') / 'data.mdb']:
            if not path.is_file():
                raise ValueError(f'Missing input: {path}')
        catalog['datasets'][record['dataset_id']] = {
            'label': f"{record['day']} · {int(record['spacing_m'] * 1000)} mm · {prefix.name.split('_')[4]}:{prefix.name.split('_')[5]} · {record['neural_role']}",
            'path': str(prefix), 'precompute_cache': str(cache),
            'empirical': str(args.empirical.resolve()), 'segmentation_version': 3.7,
            'raw_rx': record['raw_rx'], 'raw_tx': record['raw_tx'],
            'day': record['day'], 'spacing_m': record['spacing_m'], 'carrier_hz': record['carrier'],
        }
    catalog['models'][args.model_id] = {'checkpoint': str(args.checkpoint.resolve()), 'label': 'August 2026 paired model'}
    catalog.setdefault('cohorts', {}).update(manifest['cohorts'])
    from spf.inference_service.service import load_catalog
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(catalog, indent=2) + '\n')
    try:
        load_catalog(temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {args.output}: {len(manifest['records'])} frozen recordings and {len(manifest['cohorts'])} named cohorts")


if __name__ == '__main__':
    main()
