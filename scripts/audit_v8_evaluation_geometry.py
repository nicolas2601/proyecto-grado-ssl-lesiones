"""Audit reserved evaluation ZIP contents without extracting or learning from masks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from PIL import Image


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def index_members(archive, masks):
    indexed, duplicates, ignored = {}, [], 0
    for member in archive.infolist():
        name = Path(member.filename).name
        suffix = '_segmentation.png' if masks else '.jpg'
        if member.is_dir() or not name.endswith(suffix):
            ignored += 1
            continue
        identifier = name[:-len(suffix)]
        if not identifier.startswith('ISIC_') or not identifier[5:].isdigit():
            raise ValueError(f'Unexpected evaluation identifier: {member.filename}')
        if identifier in indexed:
            duplicates.append({'image_id': identifier, 'member': member.filename})
        else:
            indexed[identifier] = member
    return indexed, duplicates, ignored


def inspect_member(archive, member, mask):
    # Bound compressed-byte buffering; Pillow still loads the decoded image into RAM.
    # No archive pathname is extracted.
    checksum = hashlib.sha256()
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as spool:
        with archive.open(member) as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                checksum.update(block)
                spool.write(block)
        spool.seek(0)
        with Image.open(spool) as image:
            image.load()
            result = {'member': member.filename, 'sha256': checksum.hexdigest(),
                      'size_bytes': member.file_size, 'format': image.format,
                      'mode': image.mode, 'dimensions': list(image.size)}
            if image.format != ('PNG' if mask else 'JPEG'):
                raise ValueError(f'Unexpected image format: {image.format}')
            if mask:
                if image.mode not in ('L', '1'):
                    raise ValueError(f'Mask must be monochrome L/1: {image.mode}')
                histogram = image.convert('L').histogram()
                values = [value for value, count in enumerate(histogram) if count]
                if not (set(values).issubset({0, 1}) or set(values).issubset({0, 255})):
                    raise ValueError(f'Nonbinary mask values: {values}')
                result['values'] = values
                foreground = sum(histogram[1:])
                result['foreground_pixels'] = foreground
                result['review_findings'] = (
                    ['empty_mask'] if foreground == 0 else
                    ['full_mask'] if foreground == image.width * image.height else [])
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-zip', type=Path, required=True)
    parser.add_argument('--masks-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=2594)
    parser.add_argument('--images-sha256')
    parser.add_argument('--masks-sha256')
    args = parser.parse_args(argv)
    if args.expected_count <= 0:
        parser.error('--expected-count must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'stage': 'evaluation_geometry_audit_only',
              'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'failed',
              'script_sha256': digest(__file__), 'expected_count': args.expected_count,
              'command': [sys.executable, str(Path(__file__).resolve()),
                          *(sys.argv[1:] if argv is None else argv)],
              'inputs': {}, 'errors': [], 'review_findings': [],
              'execution_enabled': False, 'launch_ready': False, 'training_started': False,
              'manifest_frozen': False, 'training_consumable': False,
              'patient_lesion_independence': 'unresolved', 'isolation_global_verified': False,
              'limitations': ['No acredita independencia por paciente/lesión ni aislamiento global.',
                              'Las máscaras se abren exclusivamente para auditoría; no se extraen.']}
    records_path = args.output_dir / 'geometry.records.jsonl'
    checked, passed = 0, 0
    try:
        for label, path, expected in [('images', args.images_zip, args.images_sha256),
                                      ('masks', args.masks_zip, args.masks_sha256)]:
            actual = digest(path)
            report['inputs'][label] = {'path': str(path.resolve()), 'sha256': actual,
                                       'size_bytes': path.stat().st_size,
                                       'expected_sha256': expected,
                                       'expected_sha256_matches': actual == expected if expected else None}
            if expected and actual != expected:
                raise ValueError(f'{label} ZIP SHA-256 differs from expected value')
        with zipfile.ZipFile(args.images_zip) as images, zipfile.ZipFile(args.masks_zip) as masks:
            image_index, image_duplicates, image_ignored = index_members(images, False)
            mask_index, mask_duplicates, mask_ignored = index_members(masks, True)
            report['membership'] = {'image_count': len(image_index), 'mask_count': len(mask_index),
                'image_duplicates': image_duplicates, 'mask_duplicates': mask_duplicates,
                'ignored_image_members': image_ignored, 'ignored_mask_members': mask_ignored,
                'missing_masks': sorted(image_index.keys() - mask_index.keys()),
                'missing_images': sorted(mask_index.keys() - image_index.keys())}
            if (image_duplicates or mask_duplicates or image_index.keys() != mask_index.keys()
                    or len(image_index) != args.expected_count):
                raise ValueError('Evaluation ZIP membership differs from expected unique paired cohort')
            with records_path.open('x', encoding='utf-8', newline='\n') as output:
                for identifier in sorted(image_index):
                    checked += 1
                    record = {'image_id': identifier, 'status': 'failed'}
                    try:
                        record['image'] = inspect_member(images, image_index[identifier], False)
                        record['mask'] = inspect_member(masks, mask_index[identifier], True)
                        if record['image']['dimensions'] != record['mask']['dimensions']:
                            raise ValueError('Original image/mask dimensions differ')
                        record['status'] = 'passed'
                        passed += 1
                        for finding in record['mask']['review_findings']:
                            report['review_findings'].append({'image_id': identifier, 'finding': finding})
                    except Exception as exc:
                        record['error'] = f'{type(exc).__name__}: {exc}'
                        report['errors'].append({'image_id': identifier, 'error': record['error']})
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                output.flush()
                os.fsync(output.fileno())
        if passed == args.expected_count and not report['errors']:
            report['status'] = 'passed'
    except Exception as exc:
        report['errors'].append({'error': f'{type(exc).__name__}: {exc}'})
    report['checked_count'] = checked
    report['passed_count'] = passed
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    if records_path.exists():
        report['records'] = {'path': str(records_path.resolve()), 'sha256': digest(records_path)}
    temporary = args.output_dir / 'geometry.audit.json.tmp'
    final = args.output_dir / 'geometry.audit.json'
    with temporary.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, final)
    print(json.dumps({'status': report['status'], 'report': str(final.resolve()),
                      'sha256': digest(final), 'checked_count': checked,
                      'passed_count': passed, 'error_count': len(report['errors'])}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
