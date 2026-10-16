from __future__ import annotations
"""
Shared second-tier caches for several generator processes on one host (one process per GPU).

A pool is one mmap'd file (tmpfs or a disk file) holding fixed-size slots plus a slot table. Every process
maps the same file. The pool memory is never registered with HIP: GPU transfers go through a small pinned
staging buffer private to each process, and the CPU copies between staging and the pool. KFD therefore
counts only the staging buffers, and a slot is never the source or target of an in-flight DMA, so the only
cross-process rule is that slot contents and the table change under one flock.

KVPool replaces CPUPageCache (same interface, plus fetch() may return None on a miss). SharedRecurrentCache
extends RecurrentCache with the pool as a write-through second level.
"""

import fcntl
import hashlib
import mmap
import os
import queue
import struct
import threading
import time
import heapq
from collections import OrderedDict

import numpy as np
import torch

from ..constants import PAGE_SIZE
from ..cache.recurrent import RecurrentCache, note_freed

MAGIC = 0x4C4F4F5056584B58  # "XKVPOOL" + version byte space
VERSION = 1
HEADER_SIZE = 4096
_HDR = struct.Struct("<QIIQ32sQQ")  # magic, version, n_slots, slot_size, fingerprint, gen, clock
_GEN_OFF = 8 + 4 + 4 + 8 + 32
_CLOCK_OFF = _GEN_OFF + 8

ST_EMPTY, ST_VALID = 0, 1
SLOT_DTYPE = np.dtype([
    ("state", "<u4"),
    ("has_prev", "<u4"),
    ("key", "V16"),
    ("prev", "V16"),
    ("last_use", "<u8"),
    ("reserved_until", "<f8"),
    ("aux", "<i8"),        # free per-pool field (recurrent: checkpoint position)
])
assert SLOT_DTYPE.itemsize == 64

RAM_HEADROOM = 6 * 1024**3


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def _mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _log(msg: str):
    print(f" -- kvpool [{os.getpid()}]: {msg}", flush = True)


class SlotPool:
    """
    Fixed-size slots in a shared file, with a slot table in the file header. All state changes happen
    under an exclusive flock on the file, which the kernel releases if the holder dies. Readers copy slot
    contents out while holding the lock, so a reader never sees a torn slot.
    """

    def __init__(self, base: str, slot_size: int, max_size: int, fingerprint: bytes):
        self.slot_size = _align(slot_size, 4096)
        self.n_slots = int(max_size) // self.slot_size
        assert self.n_slots >= 2, f"pool {base}: {max_size} bytes is less than two slots of {self.slot_size}"
        self.fingerprint = hashlib.blake2b(
            fingerprint + struct.pack("<QQ", self.n_slots, self.slot_size), digest_size = 32
        ).digest()
        # The file name carries the fingerprint, so processes with a different model, cache layout or pool
        # size never open the same file (resetting a file another process has mapped would crash it)
        self.path = path = f"{base}.{self.fingerprint[:6].hex()}.pool"
        self.table_off = HEADER_SIZE
        self.data_off = _align(self.table_off + self.n_slots * SLOT_DTYPE.itemsize, 4096)
        self.total = self.data_off + self.n_slots * self.slot_size
        self.tlock = threading.Lock()  # flock is per open file, so threads of this process share it
        self.depth = 0                 # nesting of locked() sections in the holding thread
        self.owner = None
        self.disabled = False

        os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
        # Lease: every process using the file holds a shared lock on <file>.lease for its lifetime, so stale
        # pool files can be told apart from live ones
        self.lease_fd = os.open(path + ".lease", os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.lease_fd, fcntl.LOCK_SH)
        _remove_stale(base, keep = path)
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        with self._flock():
            st = os.fstat(self.fd)
            reuse = st.st_size == self.total and self._read_header_raw() == self._header_bytes()
            if not reuse:
                self._check_space(st)
                os.ftruncate(self.fd, 0)
                os.ftruncate(self.fd, self.total)
            self.mm = mmap.mmap(self.fd, self.total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            self.buf = np.frombuffer(self.mm, dtype = np.uint8)
            self.table = np.frombuffer(self.mm, dtype = SLOT_DTYPE, count = self.n_slots, offset = self.table_off)
            self.gen = np.frombuffer(self.mm, dtype = "<u8", count = 2, offset = _GEN_OFF)  # [gen, clock]
            if not reuse:
                self.mm[:HEADER_SIZE] = bytes(HEADER_SIZE)
                self.table[:] = np.zeros(self.n_slots, dtype = SLOT_DTYPE)
                self.mm[:_HDR.size] = self._header_bytes() + bytes(16)
            n_valid = int((self.table["state"] == ST_VALID).sum())
        _log(f"{'reused' if reuse else 'created'} {path}: {self.n_slots} slots x "
             f"{self.slot_size / 1024**2:.1f} MiB, {n_valid} valid")

        # Local mirror of the table: key -> (slot, prev_hash, aux). Refreshed when gen moves, rate limited
        self._mirror = {}
        self._mirror_gen = -1
        self._mirror_time = 0.0
        self.refresh(force = True)

    # --- file and lock

    def _header_bytes(self) -> bytes:
        return _HDR.pack(MAGIC, VERSION, self.n_slots, self.slot_size, self.fingerprint, 0, 0)[:_GEN_OFF]

    def _read_header_raw(self) -> bytes:
        return os.pread(self.fd, _GEN_OFF, 0)

    def _check_space(self, st):
        is_tmpfs = self.path.startswith("/dev/shm/")
        need = self.total - st.st_blocks * 512
        if is_tmpfs:
            avail = _mem_available()
            if avail - need < RAM_HEADROOM:
                raise MemoryError(f"pool {self.path} needs {need / 1024**3:.1f} GB but only "
                                  f"{avail / 1024**3:.1f} GB RAM is available (headroom 6 GB)")
        else:
            vfs = os.statvfs(os.path.dirname(self.path) or ".")
            if vfs.f_bavail * vfs.f_frsize < need:
                raise OSError(f"pool {self.path} needs {need / 1024**3:.1f} GB of free disk")

    class _Locked:
        """Exclusive section: thread lock, then flock. Re-entrant in the holding thread; only the outermost
        section takes and drops the flock."""
        def __init__(self, pool): self.pool = pool
        def __enter__(self):
            p = self.pool
            if p.depth and p.owner == threading.get_ident():
                p.depth += 1
                return p
            p.tlock.acquire()
            fcntl.flock(p.fd, fcntl.LOCK_EX)
            p.owner = threading.get_ident()
            p.depth = 1
            return p
        def __exit__(self, *a):
            p = self.pool
            p.depth -= 1
            if p.depth == 0:
                p.owner = None
                fcntl.flock(p.fd, fcntl.LOCK_UN)
                p.tlock.release()

    def _flock(self):
        return SlotPool._Locked(self)

    def locked(self):
        """Exclusive section. Verifies the header still matches (another process may have reset the pool for a
        different model); on mismatch the pool disables itself and every later call is a miss."""
        lk = self._flock()
        lk.__enter__()
        if not self.disabled and bytes(self.mm[:_GEN_OFF]) != self._header_bytes():
            self.disabled = True
            _log(f"{self.path} header changed under us; pool disabled for this process")
        return lk

    def _bump(self):
        self.gen[0] += 1

    def _tick(self) -> int:
        self.gen[1] += 1
        return int(self.gen[1])

    # --- mirror

    def refresh(self, force = False):
        g = int(self.gen[0])
        if not force and (g == self._mirror_gen or time.monotonic() - self._mirror_time < 0.05):
            return
        lk = self.locked()
        try:
            t = self.table.copy()
            g = int(self.gen[0])
        finally:
            lk.__exit__()
        m = {}
        for i in np.nonzero(t["state"] == ST_VALID)[0]:
            r = t[i]
            m[r["key"].tobytes()] = (int(i), r["prev"].tobytes() if r["has_prev"] else None, int(r["aux"]))
        self._mirror = m
        self._mirror_gen = g
        self._mirror_time = time.monotonic()

    def mirror(self) -> dict:
        self.refresh()
        return self._mirror

    # --- slot access, all under locked()

    def slot_view(self, slot: int) -> np.ndarray:
        o = self.data_off + slot * self.slot_size
        return self.buf[o : o + self.slot_size]

    def find(self, key: bytes) -> int | None:
        hit = self._mirror.get(key)
        if hit is not None and self.table[hit[0]]["state"] == ST_VALID and \
                self.table[hit[0]]["key"].tobytes() == key:
            return hit[0]
        idx = np.nonzero((self.table["state"] == ST_VALID) & (self.table["key"] == np.void(key)))[0]
        return int(idx[0]) if len(idx) else None

    def touch(self, slot: int):
        self.table["last_use"][slot] = self._tick()

    def reserve(self, slot: int, seconds: float):
        self.table["reserved_until"][slot] = time.time() + seconds

    def claim(self, protect: set | None, live = None) -> tuple[int, bool]:
        """Slot to overwrite: an empty one, else the first valid slot in eviction order (see _eviction_order)
        that is neither protected by the caller nor reserved by another process. Marks it empty.
        live(key) -> bool tells if a page is live in this process's GPU page table. Returns (slot, evicted)."""
        st = self.table["state"]
        empty = np.nonzero(st != ST_VALID)[0]
        if len(empty):
            return int(empty[0]), False
        now = time.time()
        first = None
        for s, k in self._eviction_order(live):
            if first is None:
                first = s
            if self.table["reserved_until"][s] > now or (protect and k in protect):
                continue
            first = s
            break
        # (if everything is protected or reserved, the first candidate goes anyway)
        self.table["state"][first] = ST_EMPTY
        self._bump()
        return first, True

    def _eviction_order(self, live):
        """Valid slots in eviction order, as in CPUPageCache: orphaned chains first, then whole trees, least
        recently used root first, each pruned tail-first, so a chain loses its newest pages before its prefix.
        A chain is orphaned when its parent is in neither the pool nor this process's GPU, and this process
        stored it (another card's GPU is not visible, so its pages count as rooted). Built fresh on each
        eviction and consumed lazily: the table is shared, so a snapshot goes stale."""
        t = self.table
        valid = np.nonzero(t["state"] == ST_VALID)[0].tolist()
        keys, prevs, has_prev = t["key"], t["prev"], t["has_prev"]
        aux, lu = t["aux"], t["last_use"]
        key = {i: keys[i].tobytes() for i in valid}
        by_key = {k: i for i, k in key.items()}
        parent = {}
        children = {i: [] for i in valid}
        orphans, roots = [], []
        me = os.getpid()
        for i in valid:
            pk = prevs[i].tobytes() if has_prev[i] else None
            pi = by_key.get(pk) if pk is not None else None
            if pi is not None and pi != i:
                parent[i] = pi
                children[pi].append(i)
            elif pk is not None and int(aux[i]) == me and live is not None and not live(pk):
                orphans.append(i)
            else:
                roots.append(i)
        for root in sorted(orphans, key = lambda i: int(lu[i])) + sorted(roots, key = lambda i: int(lu[i])):
            remaining, heap, stack = {}, [], [root]
            while stack:
                i = stack.pop()
                remaining[i] = len(children[i])
                if not children[i]:
                    heapq.heappush(heap, (int(lu[i]), i))
                stack.extend(children[i])
            while heap:
                _, i = heapq.heappop(heap)
                yield i, key[i]
                pi = parent.get(i)
                if pi is not None and pi in remaining:
                    remaining[pi] -= 1
                    if remaining[pi] == 0:
                        heapq.heappush(heap, (int(lu[pi]), pi))

    def publish(self, slot: int, key: bytes, prev: bytes | None, aux: int = 0):
        r = self.table[slot:slot + 1]
        r["key"] = np.void(key)
        r["prev"] = np.void(prev if prev is not None else bytes(16))
        r["has_prev"] = 1 if prev is not None else 0
        r["aux"] = aux
        r["reserved_until"] = 0.0
        r["last_use"] = self._tick()
        r["state"] = ST_VALID
        self._bump()
        self._mirror[key] = (slot, prev, aux)

    def close(self):
        self.table = self.gen = self.buf = None
        try:
            self.mm.close()
        except BufferError:
            pass  # views still alive; the mapping goes when they do
        os.close(self.fd)
        os.close(self.lease_fd)


def _remove_stale(base: str, keep: str):
    """Delete pool files under base that no live process holds a lease on."""
    d, prefix = os.path.split(base)
    d = d or "."
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if not name.startswith(prefix + ".") or not name.endswith(".pool") or p == keep:
            continue
        try:
            fd = os.open(p + ".lease", os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        for q in (p, p + ".lease"):
            try:
                os.unlink(q)
            except FileNotFoundError:
                pass
        os.close(fd)
        _log(f"removed stale pool {p}")


class KVPool:
    """
    Drop-in for CPUPageCache backed by a SlotPool shared between processes. A page image is the token IDs
    followed by the page's slice of every paged tensor of the attached caches (main and draft).

    store(): D2H into a pinned staging slot on the current stream; a worker thread waits for the copy, then
    writes the pool slot under the lock and publishes it. Until then the page is "pending" and is restored
    from staging directly. fetch(): copies pool -> staging under the lock (synchronous), then H2D on the
    current stream. Returns None if the page is gone, which the patched PageTable treats as a miss.
    """

    def __init__(self, caches: list, path: str, max_size: int, staging_size: int = 256 * 1024**2,
                 tag: str = ""):
        self.segments = []
        offset = PAGE_SIZE * 8  # token IDs as int64
        desc = [f"v{VERSION}", tag, f"page{PAGE_SIZE}"]
        for cache in caches:
            m = getattr(cache, "model", None)
            assert m is None or not m.loaded_tp, "KVPool does not support tensor-parallel models"
            for lk, layer in cache.layers.items():
                for t in layer.get_tensors():
                    if t is None:
                        continue
                    assert t.device.type == "cuda"
                    nbytes = t[0].numel() * t.element_size()
                    self.segments.append((t, offset, nbytes, tuple(t.shape[1:]), t.dtype))
                    desc.append(f"{tuple(t.shape)}:{t.dtype}")
                    offset = _align(offset + nbytes, 256)
        assert self.segments, "No paged cache layers for KVPool"
        self.image_size = _align(offset, 4096)
        self.pool = SlotPool(path, self.image_size, max_size, "|".join(desc).encode())
        self.max_slots = self.pool.n_slots
        self.pagetable = None

        # Write-through: job.py also pushes each page once it is complete and final, not only on eviction,
        # so the other card finds a conversation that is still live here. EXL3_KVPOOL_WT=0 turns it off
        self.write_through = os.environ.get("EXL3_KVPOOL_WT", "1") != "0"
        n_stage = max(4, staging_size // self.image_size)
        self.stage = torch.empty((n_stage, self.image_size), dtype = torch.uint8, pin_memory = True)
        self.stage_np = self.stage.numpy()
        self.stage_free = list(range(n_stage))
        self.stage_read_ev = [None] * n_stage   # event of the last H2D reading the slot
        self.stage_cond = threading.Condition()

        self.pending = {}   # phash -> (stage_idx, prev_hash, tokens)
        self.queue = queue.Queue()
        self.metrics = {
            "pushes": 0, "dedup_hits": 0, "restores": 0, "evictions": 0, "cold_allocs": 0,
            "fetch_misses": 0, "restores_pending": 0,
        }
        self._stop = False
        self._worker = threading.Thread(target = self._work, daemon = True, name = "kvpool")
        self._worker.start()

    # --- interface used by PageTable / Generator

    def attach(self, pagetable):
        self.pagetable = pagetable

    def _live(self, phash: bytes) -> bool:
        pt = self.pagetable
        return pt is not None and pt.get_live_page(phash) is not None

    def __contains__(self, phash: bytes):
        return phash in self.pending or (not self.pool.disabled and phash in self.pool.mirror())

    def __len__(self):
        return len(self.pool.mirror()) + len(self.pending)

    @property
    def entries(self):
        m = {h: {"slot": s, "prev_hash": p} for h, (s, p, _) in self.pool.mirror().items()}
        for h, (_, p, _) in list(self.pending.items()):
            m[h] = {"slot": -1, "prev_hash": p}
        return m

    # --- staging

    def _take_stage(self) -> int:
        with self.stage_cond:
            while not self.stage_free:
                self.stage_cond.wait()
            i = self.stage_free.pop()
        ev = self.stage_read_ev[i]
        if ev is not None:
            ev.synchronize()
            self.stage_read_ev[i] = None
        return i

    def _give_stage(self, i: int, read_ev = None):
        # Called with the slot exclusively ours. A pending-restore H2D may have set stage_read_ev[i] while the
        # slot held a pending store; that event is kept unless this caller has a newer one
        with self.stage_cond:
            if read_ev is not None:
                self.stage_read_ev[i] = read_ev
            self.stage_free.append(i)
            self.stage_cond.notify()

    def _views(self, buf: torch.Tensor):
        for t, off, nbytes, shape, dtype in self.segments:
            yield t, buf[off : off + nbytes].view(dtype).view(shape)

    # --- store

    def store(self, page, serial: int, protect: set | None = None):
        h = page.phash
        if h in self.pending:
            self.metrics["dedup_hits"] += 1
            return
        if h in self.pool.mirror():
            lk = self.pool.locked()
            try:
                s = self.pool.find(h)
                if s is not None:
                    self.pool.touch(s)
            finally:
                lk.__exit__()
            if s is not None:
                self.metrics["dedup_hits"] += 1
                return
        i = self._take_stage()
        buf = self.stage[i]
        tokens = page.sequence.reshape(-1).to(torch.long).clone()
        buf[: PAGE_SIZE * 8].copy_(tokens.view(torch.uint8))
        for t, v in self._views(buf):
            v.copy_(t[page.page_index], non_blocking = True)
        ev = torch.cuda.Event()
        ev.record()
        self.pending[h] = (i, page.prev_hash, tokens)
        self.queue.put((h, i, ev, page.prev_hash, set(protect) if protect else None))
        self.metrics["pushes"] += 1

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            h, i, ev, prev, protect = item
            try:
                ev.synchronize()
                lk = self.pool.locked()
                try:
                    if not self.pool.disabled and self.pool.find(h) is None:
                        s, evicted = self.pool.claim(protect, self._live)
                        self.pool.slot_view(s)[: self.image_size] = self.stage_np[i]
                        self.pool.publish(s, h, prev, aux = os.getpid())
                        if evicted:
                            self.metrics["evictions"] += 1
                finally:
                    lk.__exit__()
            except Exception as e:
                _log(f"store failed: {e!r}")
            finally:
                # Drop the pending entry and free the slot atomically with respect to a pending-restore, so a
                # restore either sees the entry (and records its event first) or goes to the pool
                with self.stage_cond:
                    self.pending.pop(h, None)
                    self.stage_free.append(i)
                    self.stage_cond.notify()

    # --- fetch

    def fetch(self, phash: bytes, page_index: int, serial: int) -> dict | None:
        with self.stage_cond:
            p = self.pending.get(phash)
            if p is not None:
                i, prev, tokens = p
                for t, v in self._views(self.stage[i]):
                    t[page_index].copy_(v, non_blocking = True)
                ev = torch.cuda.Event()
                ev.record()
                self.stage_read_ev[i] = ev
        if p is not None:
            self.metrics["restores"] += 1
            self.metrics["restores_pending"] += 1
            return {"tokens": tokens.view(1, -1), "prev_hash": prev}

        if self.pool.disabled:
            return None
        i = self._take_stage()
        ok = False
        try:
            lk = self.pool.locked()
            try:
                s = None if self.pool.disabled else self.pool.find(phash)
                if s is not None:
                    self.stage_np[i] = self.pool.slot_view(s)[: self.image_size]
                    r = self.pool.table[s]
                    prev = r["prev"].tobytes() if r["has_prev"] else None
                    self.pool.touch(s)
            finally:
                lk.__exit__()
            if s is None:
                self.metrics["fetch_misses"] += 1
                return None
            buf = self.stage[i]
            tokens = buf[: PAGE_SIZE * 8].view(torch.long).clone()
            # Slot integrity: the stored tokens must hash to the key, or the slot is not this page
            if _page_hash(tokens, prev) != phash:
                self.metrics["fetch_misses"] += 1
                _log("slot hash mismatch; treated as a miss")
                return None
            for t, v in self._views(buf):
                t[page_index].copy_(v, non_blocking = True)
            ev = torch.cuda.Event()
            ev.record()
            ok = True
            self.metrics["restores"] += 1
            return {"tokens": tokens.view(1, -1), "prev_hash": prev}
        finally:
            self._give_stage(i, ev if ok else None)

    def flush(self):
        """Wait until every pending store is in the pool (tests, shutdown)."""
        while self.pending:
            time.sleep(0.001)

    def close(self):
        self.flush()
        self.queue.put(None)
        self._worker.join(timeout = 10)
        self.segments = []
        self.pagetable = None
        self.pool.close()


def _page_hash(tokens: torch.Tensor, prev_hash: bytes | None) -> bytes:
    # Same as pagetable._tensor_blake2b_checksum over the page's (1, PAGE_SIZE) sequence tensor
    from .pagetable import tensor_hash_checksum
    return tensor_hash_checksum(tokens.view(1, -1), prev_hash)


# --------------------------------------------------------------------------------------------------------------

def _flatten(stashed: dict):
    """Ordered list of (key, index, tensor) for the tensors of a stashed recurrent state."""
    out = []
    for k in sorted((k for k in stashed if k not in ("position", "checkpoint_size", "tp_handle")), key = str):
        v = stashed[k]
        if isinstance(v, torch.Tensor):
            out.append((k, None, v))
        else:
            for j, t in enumerate(v):
                out.append((k, j, t))
    return out


class SharedRecurrentCache(RecurrentCache):
    """
    RecurrentCache (a small local LRU of stashed states) with a shared SlotPool as a write-through second
    level. put() writes new checkpoints to the pool; get() reports checkpoints that exist only in the pool
    (and reserves them so another process cannot evict them before the matching get_stashed()); get_stashed()
    loads them into fresh local tensors.
    """

    RESERVE_SECONDS = 60.0

    def __init__(self, model, max_size: int, cache, path: str, pool_size: int, tag: str = ""):
        super().__init__(model, max_size)
        assert not model.loaded_tp, "SharedRecurrentCache does not support tensor-parallel models"
        # Layout from a stash of slot 0: shapes and dtypes are the same for every checkpoint
        probe = {k: l.stash(0) for k, l in cache.get_all_recurrent_layers().items()}
        self.layout = []
        off = 64
        desc = [f"rc-v{VERSION}", tag]
        for k, j, t in _flatten(probe):
            nbytes = t.numel() * t.element_size()
            self.layout.append((k, j, off, nbytes, tuple(t.shape), t.dtype))
            desc.append(f"{k}:{j}:{tuple(t.shape)}:{t.dtype}")
            off = _align(off + nbytes, 256)
        del probe
        self.image_size = _align(off, 4096)
        self.pool = SlotPool(path, self.image_size, pool_size, "|".join(desc).encode())
        self.metrics.update({"pool_puts": 0, "pool_loads": 0, "pool_misses": 0})

    def _local_has(self, key) -> bool:
        return OrderedDict.__contains__(self, key)

    def put(self, key, state):
        super().put(key, state)
        if self.pool.disabled or not self._local_has(key):
            return
        stashed = OrderedDict.__getitem__(self, key)
        lk = self.pool.locked()
        try:
            if self.pool.disabled:
                return
            s = self.pool.find(key)
            if s is not None:
                self.pool.touch(s)
                return
            s, _ = self.pool.claim(None)
            view = self.pool.slot_view(s)
            for (k, j, off, nbytes, shape, dtype), (_, _, t) in zip(self.layout, _flatten(stashed)):
                view[off : off + nbytes] = t.contiguous().view(torch.uint8).reshape(-1).numpy()
            view[:16] = np.frombuffer(struct.pack("<qq", stashed["position"], stashed["checkpoint_size"]),
                                      dtype = np.uint8)
            self.pool.publish(s, key, None, aux = stashed["position"])
            self.metrics["pool_puts"] += 1
        finally:
            lk.__exit__()

    def get(self, key, default = None):
        v = OrderedDict.get(self, key)
        if v is not None:
            return v
        if self.pool.disabled or key not in self.pool.mirror():
            return default
        lk = self.pool.locked()
        try:
            s = self.pool.find(key)
            if s is not None:
                self.pool.reserve(s, self.RESERVE_SECONDS)
        finally:
            lk.__exit__()
        return {"shared": True} if s is not None else default

    def get_stashed(self, key, default = None):
        if self._local_has(key):
            return super().get_stashed(key, default)
        if self.pool.disabled or key not in self.pool.mirror():
            return default
        tensors = [torch.empty(shape, dtype = dtype) for (_, _, _, _, shape, dtype) in self.layout]
        lk = self.pool.locked()
        try:
            s = self.pool.find(key)
            if s is not None:
                view = self.pool.slot_view(s)
                for t, (_, _, off, nbytes, _, _) in zip(tensors, self.layout):
                    t.view(torch.uint8).reshape(-1).numpy()[:] = view[off : off + nbytes]
                position, size = struct.unpack("<qq", bytes(view[:16]))
                self.pool.touch(s)
                self.pool.table["reserved_until"][s] = 0.0
        finally:
            lk.__exit__()
        if s is None:
            self.metrics["pool_misses"] += 1
            return default
        stashed = {"position": position, "checkpoint_size": size}
        for t, (k, j, _, _, _, _) in zip(tensors, self.layout):
            if j is None:
                stashed[k] = t
            else:
                stashed.setdefault(k, []).append(t)
        for k, v in list(stashed.items()):
            if isinstance(v, list):
                stashed[k] = tuple(v)
        self._insert_local(key, stashed)
        self.metrics["pool_loads"] += 1
        return stashed

    def _insert_local(self, key, stashed):
        size = stashed["checkpoint_size"]
        while self.update_total_size() + size > self.max_size and len(self):
            _, popped = self.popitem(last = False)
            self.metrics["stash_evictions"] += 1
            note_freed(popped["checkpoint_size"])
        self[key] = stashed
        self.update_total_size()

    def close(self):
        self.pool.close()


def pool_config():
    """(kv_path, kv_bytes, rc_path, rc_bytes) from the environment, or None when the pool is off."""
    base = os.environ.get("EXL3_KVPOOL")
    if not base:
        return None
    gb = lambda n, d: int(float(os.environ.get(n, d)) * 1024**3)
    return base + ".kv", gb("EXL3_KVPOOL_GB", 8), base + ".rc", gb("EXL3_RCPOOL_GB", 4)
