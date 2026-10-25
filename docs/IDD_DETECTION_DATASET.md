# IDD-117K detection archive acquisition

The IDD-117K collection from the official IIIT Hyderabad portal supplies real-image bounding-box data for the separate detector. It does not initialize or retrain either road-segmentation model. Archive verification, extraction and preparation completed before training. See [import](IDD_DETECTION_IMPORT.md), [current results](RESULTS.md) and [final detector comparison](DETECTION_RESOLUTION_PILOT_RESULT.md).

## Downloaded archive

Five original files with suffixes `.partaa` through `.partae` form one split gzip archive. They total **72,909,856,953 compressed bytes**. Preserve their original names and order. Acquisition stores the pieces under local `dataset/idd117k_raw/`; source and data binaries are not distributed in this repository.

The completed transfer verified file sizes, byte ranges, stable server identity and local SHA-256 consistency. The full combined gzip/tar scan and extraction also passed. These are local integrity records, not publisher-authenticated checksums. Extraction matched 258,980 regular files and 73,262,082,977 unpacked bytes without creating a second concatenated archive.

Access uses expiring portal links. Private link configuration belongs only in the ignored dataset directory; tokens must not enter source, public reports or commits. URL `md5` query values authenticate access links and are not published content checksums. Obtain fresh access through the provider portal when necessary; stored ranges are reusable only when source sizes and ETags still match.

## Resumable implementation

`setup/download_idd117k.py` uses HTTP byte ranges, stable ETags, bounded concurrency and shared retry/backoff. Each finished range has a SHA-256 record. Final verification reopens ranges and records whole-part hashes. Growing files alone do not establish completion; inspect `download_verification.json` and progress status.

Initial 1/4/8-connection samples measured 0.283/1.189/1.916 MB/s. A sustained eight-connection transfer reached about 2-4 MB/s without errors. Sixteen connections triggered HTTP 429 and were stopped with completed ranges preserved; 32 were not attempted. These measurements describe that server/network period, not generally available rates.

The first sustained attempt preserved 21,072,183,296 bytes (28.9%) before exhausting its original retry limit. Subsequent bounded backoff allowed resumption while access remained valid. Authentication failure, changed source identity and local-file errors stop with evidence preserved. The downloader does not manufacture new access tokens.

Only one process may hold `.download.lock`. Creating `STOP_DOWNLOAD` in the destination requests a preserved pause; wait for `paused_preserved`. After confirming the old worker ended and clearing that marker, resume in the prepared environment:

```bash
python -u setup/download_idd117k.py --connections 8
```

A temporary wake request prevents idle sleep during an owned transfer and releases on completion; shutdown can still interrupt it. Status, range records, logs and private configuration remain local. Access controls and TLS validation must not be bypassed to resume a transfer.

## Provider membership and prepared views

The indexed provider metadata contains 96,897 annotated train identities, 20,202 annotated validation identities and 24,644 unlabelled test identities. Extra files are not assigned invented split membership. Official test images are not decoded as preprocessing examples, scored locally or used as negatives.

Recording overlap, invalid image/box pairs and exact training duplicates of validation were excluded with provenance while retained provider labels stayed in their original splits. The v2 view had 67,159 train / 17,675 validation entries. Later training-only ambiguous/repeated-pixel cleaning produced 56,729 unique v3 training images, retaining all validation membership/labels. Raw extraction and exclusions are preserved. Details are in [the import report](IDD_DETECTION_IMPORT.md) and [clean-view report](DETECTION_IMPROVEMENT.md).

The isolated runtime, annotation converters and guarded entry point were initially verified with fixtures and forward-only checks. Actual learning and final validation are distinct subsequent stages, not accomplishments of downloading the archive itself.

Official source: [IIIT Hyderabad IDD catalogue](https://insaan.iiit.ac.in/datasets/).
