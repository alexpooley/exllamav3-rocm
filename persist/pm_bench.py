"""Persistent gated-MLP prototype vs the port's kernel sequence, on real q27 layer weights (card 0).
usage: python pm_bench.py -m MODEL [--build-only] [--rows 1,8] [--layers 3,7,11,15] [--stress N]
       [--gu-splits S] [--d-splits S] [--grid G] [--prefetch 0|1]"""
import sys, os, time, argparse
TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import torch
from torch.utils.cpp_extension import load

EXT = os.path.join(TREE, "exllamav3", "exllamav3_ext")

def build():
    var = os.environ.get("PM_VARIANT", "")
    bdir = os.path.join(TREE, "persist", "build" + var); os.makedirs(bdir, exist_ok=True)
    return load(
        name="persist_mlp" + var,
        sources=[os.path.join(TREE, "persist", "persist_mlp.hip")],
        extra_include_paths=[EXT],
        extra_cuda_cflags=["-O3", "-DHIPBLAS_USE_HIP_HALF", "-mno-wavefrontsize64", "-fno-gpu-rdc",
                           "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1", "-std=c++20"] + os.environ.get("PM_FLAGS", "").split(),
        build_directory=bdir,
        verbose=False,
    )

def choose_splits(items, kblocks, target):
    splits = (target + items - 1) // items
    splits = max(1, min(splits, kblocks))
    kb = (kblocks + splits - 1) // splits
    splits = (kblocks + kb - 1) // kb
    return splits, kb * 8

class PM:
    """Scratch + launch state for one persistent MLP chain (1 or 2 layers) at M rows"""
    def __init__(self, pmx, blocks, M, args):
        self.pmx, self.blocks, self.M = pmx, blocks, M
        dev = torch.device("cuda:0")
        mlp = blocks[0].mlp
        D, I = mlp.hidden_size, mlp.intermediate_size
        self.D, self.I = D, I
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        target = 6 * nsm
        self.s_gu, self.ks_gu = choose_splits(I // 128 * 2, D // 128, target)
        self.s_d, self.ks_d = choose_splits(D // 128, I // 128, target)
        if args.gu_splits: self.s_gu, self.ks_gu = args.gu_splits, D // 16 // args.gu_splits
        if args.d_splits:
            kb = (I // 128 + args.d_splits - 1) // args.d_splits
            self.s_d, self.ks_d = (I // 128 + kb - 1) // kb, kb * 8
        occ = pmx.occupancy(M)
        self.grid = args.grid or occ * nsm
        print(f" -- M={M}: s_gu {self.s_gu} ks_gu {self.ks_gu}, s_d {self.s_d} ks_d {self.ks_d}, "
              f"occupancy {occ}/SM x {nsm} SM, grid {self.grid}", flush=True)
        z = lambda n, dt=torch.int: torch.zeros(n, dtype=dt, device=dev)
        e = lambda *s, dt=torch.float: torch.empty(*s, dtype=dt, device=dev)
        self.xh_gu = e(2 * M * D, dt=torch.half); self.xcs_gu = e(2 * M * (D // 128))
        self.c_gu = e(2, M, I, dt=torch.half); self.ws_gu = e(2 * self.s_gu * M * I); self.cnt_gu = z(2 * (I // 128))
        self.act_xh = e(M * I, dt=torch.half); self.act_xcs = e(M * (I // 128)); self.act_cnt = z(I // 128)
        self.ws_d = e(self.s_d * M * D); self.cnt_d = z(D // 128)
        self.c_d = [e(1, M, D) for _ in blocks]
        self.dep = [z(2 + self.s_d) for _ in blocks]
        self.queue = z(1); self.dummy = z(1); self.q2 = z(2)
        self.epoch = 0
        self.prefetch = args.prefetch
        self.order = args.order
        self.mode = args.mode

    def lists(self, i, xin, resid):
        b = self.blocks[i]; n = b.mlp_norm; mgu = b.mlp.multi_gu[0]; dn = b.mlp.downs[0].inner
        assert n.weight.dtype == torch.bfloat16 and mgu.K == 4 and mgu.mul1 and not mgu.mcg
        p = lambda t: t.data_ptr()
        l = [p(xin), p(resid), p(n.weight), p(mgu.ptrs_suh), p(self.xh_gu), p(self.xcs_gu), p(mgu.ptrs_trellis),
             p(mgu.ptrs_svh), p(self.c_gu), p(self.ws_gu), p(self.cnt_gu), p(dn.suh), p(self.act_xh), p(self.act_xcs),
             p(self.act_cnt), p(dn.trellis), p(dn.svh), p(self.c_d[i]), p(self.ws_d), p(self.cnt_d), p(self.dep[i])]
        f = [float(n.rms_norm_eps), float(n.constant_bias), float(n.constant_scale)]
        return l, f

    def run(self, xin, resid, part=0):
        if not part: self.epoch += 1
        l0, f0 = self.lists(0, xin, resid)
        if len(self.blocks) > 1: l1, f1 = self.lists(1, self.c_d[0], resid)
        else: l1, f1 = [], []
        cfg = [self.M, self.D, self.I, self.s_gu, self.ks_gu, self.s_d, self.ks_d, self.queue.data_ptr(), self.epoch,
               self.grid, self.prefetch, len(self.blocks), self.dummy.data_ptr(), 4,
               self.trace.data_ptr() if getattr(self, "trace", None) is not None else 0, self.order, part, self.mode, self.q2.data_ptr()]
        self.pmx.run(l0, f0, l1, f1, cfg)
        return self.c_d[-1]

def ref_chain(blocks, xin, resid):
    x = xin
    for b in blocks:
        y = b.mlp_norm.forward(x, {}, out_dtype=torch.half, residual_in=resid, had_for=b.mlp.input_bundle(resid))
        x = b.mlp.forward(y, {})
    return x

def timeit(fn, reps, inner):
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner): fn()
        b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / inner)
    ts.sort()
    return ts[len(ts) // 2], ts[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--rows", default="1,8")
    p.add_argument("--layers", default="3,7,11,15")
    p.add_argument("--stress", type=int, default=0)
    p.add_argument("--gu-splits", type=int, default=0)
    p.add_argument("--d-splits", type=int, default=0)
    p.add_argument("--grid", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=0)
    p.add_argument("--chain", type=int, default=1)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--order", type=int, default=0)
    p.add_argument("--parts", action="store_true")
    p.add_argument("--mode", type=int, default=0)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--inner", type=int, default=40)
    args = p.parse_args()
    pmx = build()
    if args.build_only:
        print("built"); return

    from exllamav3 import Config, Model
    from exllamav3.modules.transformer import TransformerBlock
    with torch.inference_mode():
        config = Config.from_directory(args.model)
        model = Model.from_config(config)
        model.load(progressbar=False)
        blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
        print(f" -- {len(blocks)} blocks", flush=True)
        D = blocks[0].mlp.hidden_size
        sel = [int(x) for x in args.layers.split(",")]
        torch.manual_seed(0)
        for M in [int(x) for x in args.rows.split(",")]:
            chains = [[blocks[i + j] for j in range(args.chain)] for i in sel]
            xin = torch.randn(1, M, D, device="cuda:0") * 0.5
            r0 = torch.randn(1, M, D, device="cuda:0") * 2.0
            pms = [PM(pmx, ch, M, args) for ch in chains]

            # correctness: bit-exact vs the port's sequence
            worst = 0.0
            for ch, pm in zip(chains, pms):
                r_ref = r0.clone(); d_ref = ref_chain(ch, xin, r_ref).clone()
                r_pm = r0.clone(); d_pm = pm.run(xin, r_pm).clone()
                torch.cuda.synchronize()
                dd = (d_ref - d_pm).abs().max().item(); dr = (r_ref - r_pm).abs().max().item()
                ex = torch.equal(d_ref, d_pm) and torch.equal(r_ref, r_pm)
                worst = max(worst, dd, dr)
                print(f"    layer {ch[0].layer_idx}+{len(ch)}: bit-exact {ex}  max|dd| {dd:.3g}  max|dr| {dr:.3g}  "
                      f"|d| {d_ref.abs().max().item():.3g}", flush=True)

            # timing, rotating over the selected layers (each MLP's weights exceed the 96 MB MALL)
            rs = [r0.clone() for _ in chains]
            def base():
                for ch, r in zip(chains, rs): ref_chain(ch, xin, r)
            def pers():
                for pm, r in zip(pms, rs): pm.run(xin, r)
            n = len(chains) * args.chain
            for fn in (base, pers): timeit(fn, 3, 10)
            tb, tbm = timeit(base, args.reps, args.inner)
            tp, tpm = timeit(pers, args.reps, args.inner)
            tb2, _ = timeit(base, args.reps, args.inner)
            mb = 3 * D * blocks[0].mlp.intermediate_size * 4 / 8 / 1e6
            floor = mb * 1e6 / 960e9 * 1e6
            print(f"  M={M} chain {args.chain}: per MLP  base {tb/n:7.1f} us (min {tbm/n:.1f}, again {tb2/n:.1f})  "
                  f"persistent {tp/n:7.1f} us (min {tpm/n:.1f})  floor {floor:.1f} us ({mb:.1f} MB)  "
                  f"-> {1 - tp/tb:+.1%}  persistent {mb*1e6/(tp/n*1e-6)/1e9:.0f} GB/s", flush=True)

            if args.parts:
                pm, r = pms[0], rs[0]
                pm.run(xin, r)
                for part, name in ((1, "gate/up"), (2, "down"), (3, "gate/up pulled by persistent grid")):
                    tq, tqm = timeit(lambda: pm.run(xin, r, part), args.reps, args.inner)
                    print(f"    part {name}: plain grid of the same unit code {tq:7.1f} us (min {tqm:.1f})", flush=True)
                # boundary + fill/drain of one gate/up launch: R passes in one persistent launch vs R launches
                m0 = pm.mode
                for R in (1, 2, 4):
                    pm.mode = R
                    tR, _ = timeit(lambda: pm.run(xin, r, 3), args.reps, args.inner)
                    pm.mode = 1
                    t1, _ = timeit(lambda: [pm.run(xin, r, 3) for _ in range(R)], args.reps, args.inner)
                    print(f"    pgu x{R}: one launch {tR:7.1f} us, {R} launches {t1:7.1f} us -> per extra launch {(t1 - tR) / max(1, R - 1):5.1f} us", flush=True)
                pm.mode = m0
                # port kernels alone, for the same split config
                b = chains[0][0]; mgu = b.mlp.multi_gu[0]
            if args.trace:
                for pm, ch in zip(pms, chains):
                    T = M + (pm.I // 128) * 2 * pm.s_gu + (pm.D // 128) * pm.s_d
                    pm.trace = torch.zeros(4 * T * len(ch), dtype=torch.long, device="cuda:0")
                    r = r0.clone()
                    for _ in range(20): pm.run(xin, r)
                    torch.cuda.synchronize()
                    t = pm.trace.view(-1, 4).cpu().double()
                    pm.trace = None
                    t0 = t[:, 0].min()
                    st, rd, en = (t[:, 0] - t0) / 100, (t[:, 1] - t0) / 100, (t[:, 2] - t0) / 100   # us
                    n_gu = (pm.I // 128) * 2 * pm.s_gu
                    for li in range(len(ch)):
                        o = li * T
                        for name, a, b in (("norm", 0, M), ("gate/up", M, M + n_gu), ("down", M + n_gu, T)):
                            sl = slice(o + a, o + b)
                            w = (rd[sl] - st[sl]).clamp(min=0) if name != "norm" else st[sl] * 0
                            body = en[sl] - torch.where(rd[sl] > 0, rd[sl], st[sl]) if name != "norm" else en[sl] - st[sl]
                            print(f"    L{li} {name:8s} units {b-a:4d}  start {st[sl].min():7.1f}..{st[sl].max():7.1f}  "
                                  f"end {en[sl].min():7.1f}..{en[sl].max():7.1f} us  wait mean {w.mean():6.1f} max {w.max():6.1f}  "
                                  f"body mean {body.mean():6.1f} min {body.min():6.1f} max {body.max():6.1f}", flush=True)
                    torch.save(t, f"trace_M{M}_c{args.chain}.pt")
                    print(f"    units per workgroup: max {torch.bincount(t[:,3].long()).max().item()}, wgs used {len(torch.unique(t[:,3]))}", flush=True)
                    break

            if args.stress:
                ch, pm = chains[0], pms[0]
                r_ref = r0.clone(); d_ref = ref_chain(ch, xin, r_ref).clone()
                bad = torch.zeros((), dtype=torch.long, device="cuda:0")
                r = r0.clone()
                t0 = time.time()
                for it in range(args.stress):
                    r.copy_(r0)
                    d = pm.run(xin, r)
                    bad += (~torch.eq(d, d_ref)).any().long() + (~torch.eq(r, r_ref)).any().long()
                    if it % 5000 == 4999:
                        torch.cuda.synchronize()
                        print(f"    stress {it+1}: mismatching launches {bad.item()}  ({time.time()-t0:.0f} s)", flush=True)
                torch.cuda.synchronize()
                print(f"  M={M} stress {args.stress} launches: mismatches {bad.item()}", flush=True)

if __name__ == "__main__":
    main()
