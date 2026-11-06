"""GDN microbench on the real q27 model: time model.forward at M rows with recurrent history on/off, the GDN
layers' share (CUDA events around each GatedDeltaNet.forward) and GDNState.rewind. Untraced unless run under
rocprofv3; with OUT given, writes OUT/window.json with each config's clock window for kernel-trace analysis.
usage: python gdnbench.py [--out OUT] [--iters N] -m MODEL -cs 16384"""
import sys, os, time, json, argparse, statistics, torch
sys.path.insert(0, os.path.join(os.environ.get("EXL3_TREE", os.path.expanduser("~/code/exl3/phoenixhaxor_exllamav3-rocm")), "examples"))

def stamps():
    return {"monotonic": time.clock_gettime_ns(time.CLOCK_MONOTONIC), "realtime": time.time_ns(),
            "boottime": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}

def main():
    from exllamav3 import model_init
    from exllamav3.modules import gated_delta_net as gdn
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--out", default=None); p.add_argument("--iters", type=int, default=60)
    p.add_argument("--configs", default="1:0,8:1,8:0,8:1r")
    args = p.parse_args()
    ev = []
    on = [False]
    f0 = gdn.GatedDeltaNet.forward
    def fw(self, x, params, out_dtype=None):
        if not on[0]: return f0(self, x, params, out_dtype)
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); r = f0(self, x, params, out_dtype); b.record(); ev.append((a, b)); return r
    gdn.GatedDeltaNet.forward = fw
    with torch.inference_mode():
        model, config, cache, tok = model_init.init(args)[:4]; print("recurrent_state history slots", next(iter(cache.get_all_recurrent_layers().values())).recurrent_state.shape, flush=True)
        P = 600
        ids = torch.randint(100, 20000, (1, P + 16))
        base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 2048)}
        prm = dict(base, past_len=0)
        model.forward(input_ids=ids[:, :P], params=prm)
        rs = prm["recurrent_states"]
        res = {}; win = {}
        for cfg in args.configs.split(","):
            m, h = cfg.split(":"); m = int(m); rew = h.endswith("r"); h = h.startswith("1")
            tf, tg, tr = [], [], []
            torch.cuda.synchronize(); time.sleep(0.05)
            for it in range(args.iters + 5):
                if it == 5: torch.cuda.synchronize(); t0 = stamps()
                ev.clear(); on[0] = it >= 5
                a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); c = torch.cuda.Event(enable_timing=True)
                a.record()
                model.forward(input_ids=ids[:, P:P + m], params=dict(base, past_len=P, recurrent_states=rs, recurrent_history=h))
                b.record()
                if rew: rs[0].rewind(m - 4)   # partial accept: keep 4 of m
                c.record()
                on[0] = False
                rs[0].position = P; rs[0].last_history = 0
                if it >= 5:
                    torch.cuda.synchronize()
                    tf.append(a.elapsed_time(b)); tr.append(b.elapsed_time(c))
                    tg.append(sum(x.elapsed_time(y) for x, y in ev))
            torch.cuda.synchronize(); t1 = stamps()
            win[cfg] = {"window": {k: [t0[k], t1[k]] for k in t0}, "iters": args.iters}
            md = statistics.median
            res[cfg] = (md(tf), md(tg), md(tr))
            print(f"M={m} history={int(h)} rewind={int(rew)}: forward {md(tf):7.3f} ms  GDN layers {md(tg):7.3f} ms ({len(ev)} layers)  rewind {md(tr)*1e3:7.1f} us", flush=True)
        if args.out:
            os.makedirs(args.out, exist_ok=True); json.dump(win, open(f"{args.out}/window.json", "w"), indent=1)

if __name__ == "__main__":
    main()
