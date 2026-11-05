"""Retrieve the versioned, CC BY 4.0 Joung et al. optical-property data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
META_URL = "https://api.figshare.com/v2/articles/12045567/versions/2"


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    meta_path = raw / "figshare_12045567_v2_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        response = requests.get(META_URL, timeout=60)
        response.raise_for_status()
        meta = response.json()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    assert meta["doi"] == "10.6084/m9.figshare.12045567.v2"
    records = []
    for file in meta["files"]:
        path = raw / file["name"]
        assert path.parent.resolve() == raw.resolve()
        if not path.exists():
            response = requests.get(file["download_url"], timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        blob = path.read_bytes()
        assert len(blob) == file["size"]
        assert hashlib.md5(blob).hexdigest() == file["computed_md5"]
        records.append({"file": path.relative_to(ROOT).as_posix(), "download_url": file["download_url"],
                        "md5": file["computed_md5"], "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
    lock = {"metadata_url": META_URL, "dataset_doi": meta["doi"], "license": meta["license"],
            "article_doi": "10.1038/s41597-020-00634-8", "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": records}
    lock_path = ROOT / "data/source_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text(encoding="utf-8"))["files"] == records
    else:
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
