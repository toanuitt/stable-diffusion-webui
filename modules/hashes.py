import hashlib
import json
import os.path

import filelock


cache_filename = "cache.json"
cache_data = None


def dump_cache():
    with filelock.FileLock(cache_filename+".lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)


def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(cache_filename+".lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256(filename, title):
    hashes = cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)

    if title in hashes:
        cached_sha256 = hashes[title].get("sha256", None)
        cached_mtime = hashes[title].get("mtime", 0)

        if ondisk_mtime <= cached_mtime and cached_sha256 is not None:
            return cached_sha256

    print(f"Calculating sha256 for {filename}: ", end='')
    sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    hashes[title] = {
        "mtime": ondisk_mtime,
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value




