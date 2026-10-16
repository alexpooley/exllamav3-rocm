#!/usr/bin/env python3
"""Standalone tests for exllamav3/generator/kvpool.py, without a model.
Runs worker processes on both cards (ROCR_VISIBLE_DEVICES 0 and 1) against one pool under a test base path.
usage: test_kvpool.py            (runs all tests)
       test_kvpool.py _role ...  (internal)"""
import os, subprocess, sys, time, signal, struct, random
import torch

BASE = "/dev/shm/kvpool-test/t"
PY = sys.executable
ME = os.path.abspath(__file__)
N_PAGES_GPU = 64
torch.manual_seed(0)


# ---------------------------------------------------------------- fakes

class FakeLayer:
    def __init__(self, n, dev):
        self.k = torch.zeros((n, 256, 2, 64), dtype = torch.bfloat16, device = dev)
        self.v = torch.zeros((n, 256, 2, 64), dtype = torch.bfloat16, device = dev)
    def get_tensors(self):
        return [self.k, self.v]

class FakeCache:
    def __init__(self, n = N_PAGES_GPU, dev = "cuda"):
        self.model = None
        self.layers = {0: FakeLayer(n, dev), 1: FakeLayer(n, dev)}

class FakePageTable:
    """No page is live on the GPU, so a stored page whose parent is gone is an orphan."""
    def get_live_page(self, phash):
        return None

class FakePage:
    def __init__(self, seq, prev, page_index):
        from exllamav3.generator.pagetable import tensor_hash_checksum
        self.sequence = seq.view(1, -1)
        self.prev_hash = prev
        self.phash = tensor_hash_checksum(self.sequence, prev)
        self.page_index = page_index

def tokens_for(chain: int, i: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(chain * 100003 + i)
    return torch.randint(0, 150000, (256,), generator = g, dtype = torch.long)

def chain_pages(chain: int, n: int):
    """Hashes and token tensors of a deterministic page chain."""
    prev, out = None, []
    for i in range(n):
        p = FakePage(tokens_for(chain, i), prev, 0)
        out.append(p)
        prev = p.phash
    return out

def fill_gpu(cache, page_index, phash):
    """Write a pattern derived from the page hash into every tensor of page page_index."""
    seed = int.from_bytes(phash[:4], "little")
    for li, l in cache.layers.items():
        for ti, t in enumerate(l.get_tensors()):
            g = torch.Generator(device = "cuda").manual_seed(seed + li * 10 + ti)
            t[page_index].copy_(torch.randn(t[page_index].shape, generator = g, device = "cuda").to(t.dtype))

def check_gpu(cache, page_index, phash) -> bool:
    seed = int.from_bytes(phash[:4], "little")
    for li, l in cache.layers.items():
        for ti, t in enumerate(l.get_tensors()):
            g = torch.Generator(device = "cuda").manual_seed(seed + li * 10 + ti)
            exp = torch.randn(t[page_index].shape, generator = g, device = "cuda").to(t.dtype)
            if not torch.equal(t[page_index], exp):
                return False
    return True

def init_hip():
    torch.arange(8, device = "cuda").sum().item()  # a kernel before pinning (see ADR 0058)

def wait_file(p, timeout = 60):
    t = time.time()
    while not os.path.exists(p):
        if time.time() - t > timeout:
            raise TimeoutError(p)
        time.sleep(0.05)


# ---------------------------------------------------------------- roles (run in subprocesses)

def role_store(pool_mb, chain, n, done_flag):
    from exllamav3.generator.kvpool import KVPool
    init_hip()
    cache = FakeCache()
    tier = KVPool([cache], BASE + ".kv", pool_mb << 20, staging_size = 8 << 20, tag = "test")
    tier.attach(FakePageTable())
    for k, p in enumerate(chain_pages(chain, n)):
        p.page_index = k % N_PAGES_GPU
        fill_gpu(cache, p.page_index, p.phash)
        tier.store(p, 0)
        if k % N_PAGES_GPU == N_PAGES_GPU - 1:
            torch.cuda.synchronize()   # the engine never overwrites a page before its D2H; mimic that
    tier.flush()
    print("STORED", tier.metrics, flush = True)
    open(done_flag, "w").close()
    tier.close()

def role_fetch(pool_mb, chain, n, expect):
    """expect: 'all' every page restores; 'head:K' K pages restore, the chain head among them."""
    from exllamav3.generator.kvpool import KVPool
    init_hip()
    cache = FakeCache()
    tier = KVPool([cache], BASE + ".kv", pool_mb << 20, staging_size = 8 << 20, tag = "test")
    pages = chain_pages(chain, n)
    ok = bad = miss = 0
    head_ok = False
    for k, p in enumerate(pages):
        dst = (k * 7 + 3) % N_PAGES_GPU
        present = p.phash in tier
        e = tier.fetch(p.phash, dst, 0) if present else None
        if e is None:
            miss += 1
            continue
        torch.cuda.synchronize()
        good = check_gpu(cache, dst, p.phash) and torch.equal(e["tokens"].view(-1), p.sequence.view(-1)) \
            and e["prev_hash"] == p.prev_hash
        ok += good
        bad += not good
        head_ok |= good and k == 0
    # is_resumable-style chain walk through entries
    entries = tier.entries
    print(f"FETCH ok={ok} bad={bad} miss={miss} entries={len(entries)}", tier.metrics, flush = True)
    if expect == "all":
        assert ok == n and bad == 0, "not every page restored correctly"
    else:
        k = int(expect.split(":")[1])
        assert bad == 0 and ok == k and miss == n - k and head_ok, f"expected {k} pages including the head"
    tier.close()

def role_hold_lock(pool_mb):
    from exllamav3.generator.kvpool import KVPool
    init_hip()
    tier = KVPool([FakeCache()], BASE + ".kv", pool_mb << 20, staging_size = 8 << 20, tag = "test")
    tier.pool.locked()
    print("LOCKED", flush = True)
    time.sleep(600)

def role_try_lock(pool_mb):
    from exllamav3.generator.kvpool import KVPool
    init_hip()
    tier = KVPool([FakeCache()], BASE + ".kv", pool_mb << 20, staging_size = 8 << 20, tag = "test")
    lk = tier.pool.locked(); lk.__exit__()
    print("GOT LOCK", flush = True)
    tier.close()

def role_stress(pool_mb, seed, seconds):
    """Random store/fetch over a page universe bigger than the pool; every restore must be exact."""
    from exllamav3.generator.kvpool import KVPool
    init_hip()
    cache = FakeCache()
    tier = KVPool([cache], BASE + ".kv", pool_mb << 20, staging_size = 8 << 20, tag = "test")
    rng = random.Random(seed)
    universe = [p for c in range(6) for p in chain_pages(1000 + c, 30)]
    t_end = time.time() + seconds
    stores = fetches = ok = bad = miss = 0
    k = 0
    while time.time() < t_end:
        p = rng.choice(universe)
        if rng.random() < 0.5:
            p.page_index = k % N_PAGES_GPU
            fill_gpu(cache, p.page_index, p.phash)
            tier.store(p, 0)
            stores += 1
            k += 1
            if k % 16 == 0:
                torch.cuda.synchronize()
        else:
            dst = rng.randrange(N_PAGES_GPU)
            e = tier.fetch(p.phash, dst, 0) if p.phash in tier else None
            fetches += 1
            if e is None:
                miss += 1
                continue
            torch.cuda.synchronize()
            if check_gpu(cache, dst, p.phash) and torch.equal(e["tokens"].view(-1), p.sequence.view(-1)):
                ok += 1
            else:
                bad += 1
            # the destination page now holds this page's data; it may be stored again later from here
    tier.flush()
    print(f"STRESS seed={seed} stores={stores} fetches={fetches} ok={ok} bad={bad} miss={miss}", tier.metrics,
          flush = True)
    assert bad == 0
    tier.close()


class FakeRLayer:
    def __init__(self, dev):
        self.rs = torch.randn((2, 1, 4, 16, 16), dtype = torch.float, device = dev)
        self.cs = torch.randn((2, 32, 4), dtype = torch.bfloat16, device = dev)
    def stash(self, slot, position = 0):
        return (self.rs[slot, :1].cpu(), self.cs[slot].cpu())

class FakeRCache:
    def __init__(self, dev = "cuda"):
        self.layers = {"a": FakeRLayer(dev), "b": FakeRLayer(dev)}
    def get_all_recurrent_layers(self):
        return self.layers

class FakeModel:
    loaded_tp = False

class FakeState:
    def __init__(self, cache, position, seed):
        self.cache, self.position, self.seed = cache, position, seed
    def stash(self):
        g = torch.Generator().manual_seed(self.seed)
        d = {"position": self.position, "checkpoint_size": 0}
        size = 0
        for k, l in self.cache.layers.items():
            a = torch.randn(l.rs[0, :1].shape, generator = g)
            b = torch.randn(l.cs[0].shape, generator = g).to(torch.bfloat16)
            d[k] = (a, b)
            size += a.numel() * 4 + b.numel() * 2
        d["checkpoint_size"] = size
        return d

def role_rc(mode, n):
    from exllamav3.generator.kvpool import SharedRecurrentCache
    init_hip()
    rcache = FakeRCache()
    rc = SharedRecurrentCache(FakeModel(), 1 << 20, rcache, BASE + ".rc", 4 << 20, tag = "test")
    keys = [bytes([i + 1]) * 16 for i in range(n)]
    if mode == "put":
        for i, k in enumerate(keys):
            rc.put(k, FakeState(rcache, (i + 1) * 256, i))
        print("RC PUT", rc.metrics, flush = True)
    else:
        good = 0
        for i, k in enumerate(keys):
            if not rc.get(k):
                continue
            s = rc.get_stashed(k)
            exp = FakeState(rcache, (i + 1) * 256, i).stash()
            same = s["position"] == exp["position"] and all(
                torch.equal(s[x][0], exp[x][0]) and torch.equal(s[x][1], exp[x][1]) for x in ("a", "b"))
            good += same
            assert same, f"checkpoint {i} differs"
        print("RC GET good", good, rc.metrics, flush = True)
        assert good >= 1
    rc.close()


# ---------------------------------------------------------------- orchestration

def run(card, *args, wait = True, timeout = 300):
    env = dict(os.environ, ROCR_VISIBLE_DEVICES = str(card))
    p = subprocess.Popen([PY, ME, "_role", *map(str, args)], env = env, stdout = subprocess.PIPE,
                         stderr = subprocess.STDOUT, text = True)
    if not wait:
        return p
    out, _ = p.communicate(timeout = timeout)
    return p.returncode, "\n".join(l for l in out.splitlines() if "Warning" not in l and "warn(" not in l)

def fresh():
    subprocess.run(["rm", "-rf", os.path.dirname(BASE)])
    os.makedirs(os.path.dirname(BASE))

def check(name, cond, out = ""):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        print(out)
    return cond

def main():
    results = []
    # 1. store on card 0, restore on card 1 (and the reverse), exact bytes
    fresh()
    rc, out = run(0, "store", 64, 1, 40, BASE + ".done1")
    rc2, out2 = run(1, "fetch", 64, 1, 40, "all")
    results.append(check("card0 store -> card1 fetch, 40 pages exact", rc == 0 and rc2 == 0, out + out2))
    rc, out = run(1, "store", 64, 2, 40, BASE + ".done2")
    rc2, out2 = run(0, "fetch", 64, 2, 40, "all")
    results.append(check("card1 store -> card0 fetch, 40 pages exact", rc == 0 and rc2 == 0, out + out2))

    # 2. Eviction: a pool of ~16 slots holding a longer chain prunes tail-first, so the head stays; misses are clean
    fresh()
    slots_mb = 4   # 16 slots of ~256 KiB
    rc, out = run(0, "store", slots_mb, 3, 40, BASE + ".done3")
    import re
    m = re.search(r"'evictions': (\d+)", out)
    rc2, out2 = run(1, "fetch", slots_mb, 3, 40, f"head:{40 - int(m.group(1))}" if m else "head:0")
    results.append(check("tail-first eviction keeps the chain head, misses return None", rc == 0 and rc2 == 0,
                         out + out2))

    # 3. pool survives the writer exiting; a restart reuses the file
    rc2, out2 = run(0, "fetch", slots_mb, 3, 40, f"head:{40 - int(m.group(1))}" if m else "head:0")
    results.append(check("pool survives writer exit and is reused on reopen", rc2 == 0 and "reused" in out2, out2))

    # 4. a process killed while holding the lock does not block the other
    fresh()
    h = run(0, "hold", 4, wait = False)
    line = ""
    t = time.time()
    while "LOCKED" not in line and time.time() - t < 120:
        line = h.stdout.readline()
    os.kill(h.pid, signal.SIGKILL); h.wait()
    rc, out = run(1, "trylock", 4, timeout = 60)
    results.append(check("lock released when its holder is SIGKILLed", rc == 0 and "GOT LOCK" in out, out))

    # 5. different layout -> different file; stale files without a lease are removed
    stale = BASE + ".kv.deadbeef0000.pool"
    open(stale, "w").close()
    rc, out = run(0, "store", 8, 4, 4, BASE + ".done5")
    results.append(check("stale pool file without lease removed", rc == 0 and not os.path.exists(stale), out))

    # 6. stress: both cards store/fetch at once for 30 s over a universe 3x the pool
    fresh()
    a = run(0, "stress", 12, 1, 30, wait = False)
    b = run(1, "stress", 12, 2, 30, wait = False)
    oa, _ = a.communicate(timeout = 300); ob, _ = b.communicate(timeout = 300)
    keep = lambda o: "\n".join(l for l in o.splitlines() if l.startswith(("STRESS", "Traceback", "Assert")) or "Error" in l)
    results.append(check("concurrent stress on both cards, no corrupt restore",
                         a.returncode == 0 and b.returncode == 0, keep(oa) + "\n" + keep(ob)))
    print(keep(oa)); print(keep(ob))

    # 7. recurrent checkpoints: put on card 0, get on card 1
    fresh()
    rc, out = run(0, "rcput", "put", 8)
    rc2, out2 = run(1, "rcget", "get", 8)
    results.append(check("recurrent checkpoints shared across cards", rc == 0 and rc2 == 0, out + out2))
    print(out2.splitlines()[-1] if out2 else "")

    subprocess.run(["rm", "-rf", os.path.dirname(BASE)])
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_role":
        r, a = sys.argv[2], sys.argv[3:]
        if r == "store": role_store(int(a[0]), int(a[1]), int(a[2]), a[3])
        elif r == "fetch": role_fetch(int(a[0]), int(a[1]), int(a[2]), a[3])
        elif r == "hold": role_hold_lock(int(a[0]))
        elif r == "trylock": role_try_lock(int(a[0]))
        elif r == "stress": role_stress(int(a[0]), int(a[1]), float(a[2]))
        elif r in ("rcput", "rcget"): role_rc(a[0], int(a[1]))
    else:
        main()
