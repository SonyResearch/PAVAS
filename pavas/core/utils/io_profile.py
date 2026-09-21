import os
import threading
import time
from contextlib import contextmanager

_ENVAR = 'MMAUDIO_PROFILE_IO'

def _env_true(v: str | None) -> bool:
    if v is None: return False
    return v.lower() in {'1','true','yes','on'}

ENABLE_IO_PROFILE = _env_true(os.environ.get(_ENVAR))

_lock = threading.Lock()
_acc: dict[str,float] = {}
_cnt: dict[str,int] = {}
_hit: dict[str,dict[str,int]] = {}

def enable_io_profile(flag: bool = True):
    global ENABLE_IO_PROFILE
    ENABLE_IO_PROFILE = flag

def io_profile_snapshot(reset: bool = True):
    if not ENABLE_IO_PROFILE:
        return {}
    with _lock:
        snap = {
            k: {
                'total': _acc[k],
                'count': _cnt[k],
                'avg': (_acc[k] / max(1,_cnt[k])),
                'hit': _hit.get(k,{}).get('hit'),
                'miss': _hit.get(k,{}).get('miss'),
            } for k in _acc.keys()
        }
        if reset:
            _acc.clear(); _cnt.clear(); _hit.clear()
    return snap

def _add(key: str, dt: float, hit: bool | None = None):
    if not ENABLE_IO_PROFILE:
        return
    with _lock:
        _acc[key] = _acc.get(key, 0.0) + dt
        _cnt[key] = _cnt.get(key, 0) + 1
        if hit is not None:
            s = _hit.get(key)
            if s is None:
                s = {'hit':0,'miss':0}
                _hit[key] = s
            if hit: s['hit'] += 1
            else: s['miss'] += 1

@contextmanager
def io_timer(key: str, hit: bool | None = None):
    if not ENABLE_IO_PROFILE:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _add(key, time.perf_counter() - t0, hit=hit)

def format_snapshot(snap: dict[str,dict]) -> str:
    parts = []
    for k,v in sorted(snap.items()):
        hit = v.get('hit'); miss = v.get('miss')
        hm = ''
        if hit is not None:
            total_hm = (hit or 0) + (miss or 0)
            miss_rate = (miss / total_hm * 100.0) if total_hm else 0.0
            hm = f" hit={hit} miss={miss} miss_rate={miss_rate:.1f}%"
        parts.append(f"{k}:avg={v['avg']*1000:.2f}ms cnt={v['count']}{hm}")
    return ' | '.join(parts)

def io_add(key: str, dt: float, hit: bool | None = None):
    """Programmatic add (e.g., from aggregated worker metrics)."""
    _add(key, dt, hit=hit)

__all__ = [
    'ENABLE_IO_PROFILE','enable_io_profile','io_profile_snapshot','io_timer','format_snapshot','io_add'
]
