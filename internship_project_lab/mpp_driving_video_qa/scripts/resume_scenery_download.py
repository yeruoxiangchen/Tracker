#!/usr/bin/env python3
"""Resume the official Scenery ZIP with validated, short HTTP range requests."""
import argparse
import fcntl
from http.cookiejar import MozillaCookieJar
from pathlib import Path
import re
import time
import zipfile

import requests

URL = ('https://drive.usercontent.google.com/download'
       '?id=1Qv04SMKRMWW0sAGRTNCrXIDPbRaZWFrR&export=download&confirm=t')


def fetch_range(session, start, end):
    with session.get(URL, headers={'Range': 'bytes=%d-%d' % (start, end),
                                  'Accept-Encoding': 'identity'},
                     stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                             response.headers.get('Content-Range', ''))
        if response.status_code != 206 or not match:
            raise RuntimeError('Server did not return a partial file; nothing appended')
        actual_start, actual_end, total = map(int, match.groups())
        if actual_start != start or actual_end != min(end, total - 1):
            raise RuntimeError('Unexpected Content-Range; nothing appended')
        data = bytearray()
        for block in response.iter_content(1024 * 1024):
            data.extend(block)
            if len(data) > actual_end - start + 1:
                raise RuntimeError('Range response exceeds requested size')
        if len(data) != actual_end - start + 1:
            raise RuntimeError('Incomplete chunk; nothing appended')
        return data, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('/data/zjr/datasets/lingoqa/scenery/images.zip'))
    args = parser.parse_args()
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0'
    cookie_path = Path.home() / '.cache/gdown/cookies.txt'
    if cookie_path.exists():
        jar = MozillaCookieJar(str(cookie_path))
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies.update(jar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a+b') as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check the existing prefix before resuming; never truncate the partial ZIP.
        prefix, total = fetch_range(session, 0, 65535)
        output.seek(0, 2)
        size = output.tell()
        output.seek(0)
        if size > total or output.read(min(size, len(prefix))) != prefix[:min(size, len(prefix))]:
            raise RuntimeError('Existing file does not match the remote ZIP; left unchanged')
        output.seek(0, 2)
        print('Resuming %s: %d/%d bytes' % (args.output, size, total), flush=True)
        failures = 0
        while size < total:
            started = time.monotonic()
            try:
                data, remote_total = fetch_range(session, size, min(size + 32 * 1024**2 - 1, total - 1))
                if remote_total != total:
                    raise RuntimeError('Remote size changed')
            except (requests.RequestException, RuntimeError) as exc:
                failures += 1
                print('Retry %d/20: %s' % (failures, exc), flush=True)
                if failures >= 20:
                    raise
                time.sleep(10)
                continue
            output.write(data)
            output.flush()
            size += len(data)
            failures = 0
            print('SAVED %d/%d bytes (%.2f%%), %.2f MiB/s' %
                  (size, total, 100 * size / total,
                   len(data) / 1024**2 / max(time.monotonic() - started, .001)), flush=True)
        print('Download complete; verifying ZIP CRC...', flush=True)
        with zipfile.ZipFile(args.output) as archive:
            bad = archive.testzip()
        if bad:
            raise RuntimeError('ZIP CRC failed: %s' % bad)
        print('ZIP VERIFIED: %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
