"""q27 bare forward timing at fixed row counts (no sampler, no drafter): ms per forward for R rows.
Values do not matter (codebook timing experiment), so the input ids are fixed text.
usage: python fwd_bench.py -m MODEL -cs 16384 [--rows 1,8] [--steps 100] [--ctx 1024]"""
import sys, os, time, argparse
sys.path.insert(0, os.environ["EXL3_TREE"])
import torch
from exllamav3 import model_init

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True)
    p.add_argument("--rows", default="1,8"); p.add_argument("--steps", type=int, default=200)
    p.add_argument("--ctx", type=int, default=1024); p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    _im = torch.inference_mode(); _im.__enter__()
    model, config, cache, tok = model_init.init(args)
    import exllamav3
    print("exllamav3 from", os.path.dirname(exllamav3.__file__), flush=True)
    ids_all = tok.encode("The history of the lighthouse begins with simple fires on hilltops. " * 400)
    rec = model.caps.get("recurrent_states", False)
    st = {"pos": 0, "rec": [cache.get_test_state(0)] if rec else None}
    def params(pos, R):
        return {"attn_mode": "flash_attn", "cache": cache, "past_len": pos,
                "batch_shape": (1, args.ctx + 4096), "recurrent_states": st["rec"]}
    model.prefill(ids_all[:, :args.ctx], params(0, 1)); st["pos"] = args.ctx
    def step(R):
        pos = st["pos"]
        logits = model.forward(ids_all[:, pos:pos + R], params(pos, R))
        torch.argmax(logits[:, -1]).cpu()
        st["pos"] += R   # no wrap: the recurrent (GDN) state must follow past_len
    for R in [int(r) for r in args.rows.split(",")]:
        n = max(10, args.steps // R)
        for _ in range(10): step(R)
        res = []
        for _ in range(args.reps):
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(n): step(R)
            torch.cuda.synchronize(); res.append((time.perf_counter() - t) / n * 1e3)
        print(f"FWD rows={R} ms/forward " + " ".join(f"{v:.3f}" for v in res) + f"  min {min(res):.3f}", flush=True)

if __name__ == "__main__":
    main()
