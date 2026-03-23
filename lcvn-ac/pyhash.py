import hashlib


def fnv1_32():
    class _Hasher:
        def __call__(self, s):
            # Deterministic 32-bit integer using MD5 prefix
            h = hashlib.md5(str(s).encode("utf-8")).hexdigest()[:8]
            return int(h, 16)

    return _Hasher()