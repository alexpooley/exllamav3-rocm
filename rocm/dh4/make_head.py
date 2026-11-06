"""Build a lower-bit EXL3 copy of the target lm_head for the DFlash2 drafter (EXL3_DRAFT_HEAD).
Phase capture: load target + drafter, generate on calibration prompts (not the bench prompts), and sum
H = X^T X over the drafter states the head sees (rows 1.., as used by the selector). Phase quant: dequantise
the target's stored head (the only head weights on disk) and EXL3-quantise it to K bits with H (mul1 codebook,
out scales, as the model's own head); save lm_head.* to a safetensors file.
usage: python make_head.py capture -m MODEL -dm DRAFT -cs 16384 --work DIR
       python make_head.py quant -m MODEL --work DIR --bits 4 --out FILE"""
import sys, os, argparse, time, torch
sys.path.insert(0, os.path.join(os.environ.get("EXL3_TREE", os.path.expanduser("~/code/exl3/phoenixhaxor_exllamav3-rocm")), "examples"))

CALIB = [
    "Implement a thread-safe bounded queue in C++ with condition variables and explain each method.",
    "Write a Rust function that parses a CSV line with quoted fields, plus tests.",
    "Explain how a B-tree insert works, then write it in Go.",
    "Write a bash script that rotates log files older than 7 days and compresses them.",
    "Write a SQL schema for a library system with loans and fines, and three example queries.",
    "Refactor this JavaScript into async/await: fetch(url).then(r => r.json()).then(d => render(d)).catch(e => log(e));",
    "Return a JSON object describing three fictional employees with name, role, skills (array) and start_date.",
    "Produce a JSON array of 8 cities with fields name, country, population, lat, lon. Output only JSON.",
    "Write an OpenAPI 3 YAML spec for a todo API with CRUD endpoints.",
    "Write a 500-word essay on the history of the printing press.",
    "Describe a walk through a rainforest at dawn, in vivid sensory prose.",
    "Write a letter from a ship captain to his daughter during a storm.",
    "Summarise the causes of the French Revolution in five paragraphs.",
    "Explain photosynthesis to a ten-year-old, then to a university student.",
    "A tank fills in 6 hours with pipe A and 9 hours with pipe B; pipe C drains it in 12 hours. How long with all three open? Show steps.",
    "Prove that the square root of 2 is irrational, step by step.",
    "Compare TCP and UDP in a markdown table, then explain when to use each.",
    "Write a Python script using argparse that renames photos by their EXIF date.",
    "Give a recipe for sourdough bread with a timeline, in a numbered list.",
    "Translate into French and explain the grammar: 'I would have gone if I had known.'",
]

def capture(args):
    import exllamav3
    from exllamav3 import model_init, Generator, Job, GreedySampler
    from common import format_prompt
    with torch.inference_mode():
        model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(args)
        lm = model.modules[model.logit_layer_idx]
        print("target head:", type(lm.inner).__name__, "K", getattr(lm.inner, "K", None),
              "trellis", tuple(lm.inner.trellis.shape), lm.inner.trellis.numel() * 2 / 1e6, "MB", flush=True)
        dev = lm.device
        H = torch.zeros((lm.in_features, lm.in_features), dtype=torch.float, device=dev)
        n = [0]
        orig = dmodel.sample_from_state
        def hooked(state, params):
            x = state[:, 1:].reshape(-1, state.shape[-1]).to(dev).float()
            H.addmm_(x.T, x); n[0] += x.shape[0]
            return orig(state, params)
        dmodel.sample_from_state = hooked
        gen = Generator(model=model, cache=cache, draft_model=dmodel, draft_cache=dcache, tokenizer=tok)
        t0 = time.time()
        for i, q in enumerate(CALIB):
            ids = tok.encode(format_prompt("chatml", "You are a helpful assistant.", q, no_think=True) + "\n\n", encode_special_tokens=True)
            gen.enqueue(Job(input_ids=ids, max_new_tokens=args.tokens, stop_conditions=[], sampler=GreedySampler()))
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "error": raise r["error"]
            print(f"calib {i+1}/{len(CALIB)} rows {n[0]} {time.time()-t0:.0f}s", flush=True)
        os.makedirs(args.work, exist_ok=True)
        torch.save({"H": H.cpu(), "count": n[0]}, os.path.join(args.work, "H.pt"))

def refit_scales_inplace(weight, weight_q, H, su, sv, rounds = 2, chunk = 16384):
    """quantize.refit_scales with in-place scaling and a chunked row sum: same maths, no full-size temporaries
    (the stock version needs 4 x 5 GB for a 248k-wide head and does not fit in 24 GB)"""
    dev = weight_q.device
    H = H.to(dev)
    def cols(x, a, b):
        return (x[:, a:b] if x.device == dev else x[:, a:b].to(dev)).float()
    k, n = weight_q.shape
    Q = weight_q
    su = su.float().to(dev).view(k, 1)
    sv = sv.float().to(dev).view(1, n)
    HW = torch.empty((k, n), dtype = torch.float, device = dev)
    den = 0.0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        wc = cols(weight, a, b)
        HW[:, a:b] = H @ wc
        den += (wc * HW[:, a:b]).sum().item()
    def err(Qx):
        e = 0.0
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            E = cols(weight, a, b) - Qx[:, a:b]
            e += (E * (H @ E)).sum().item()
        return e / max(den, 1e-30)
    e_before = err(Q)
    for _ in range(rounds):
        num = torch.zeros(n, device = dev); dn = torch.zeros(n, device = dev)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            HQ = H @ Q[:, a:b]
            num[a:b] = (Q[:, a:b] * HW[:, a:b]).sum(0)
            dn[a:b] = (Q[:, a:b] * HQ).sum(0)
        c = torch.where(dn > 1e-30, num / dn.clamp(min = 1e-30), torch.ones_like(num))
        Q.mul_(c.view(1, n))
        sv = sv * c.view(1, n)
        A = (Q @ Q.T) * H
        b_ = torch.zeros(k, device = dev)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            b_ += (Q[:, a:b] * HW[:, a:b]).sum(1)
        A.diagonal().add_(1e-6 * A.diagonal().mean())
        r = torch.linalg.solve(A, b_)
        r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r))
        Q.mul_(r.view(k, 1))
        su = su * r.view(k, 1)
    e_after = err(Q)
    return Q, su, sv, e_before, e_after

def quant(args):
    import exllamav3.modules.quant.exl3_lib.quantize as qz
    qz.refit_scales = refit_scales_inplace
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from safetensors.torch import save_file
    import json
    from safetensors import safe_open
    from exllamav3.modules.quant.exl3 import LinearEXL3
    d = torch.load(os.path.join(args.work, "H.pt"))
    dev = torch.device("cuda:0")
    idx = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))["weight_map"]
    src = {}
    for k in ("suh", "svh", "trellis", "mul1", "mcg"):
        if f"lm_head.{k}" in idx:
            with safe_open(os.path.join(args.model_dir, idx[f"lm_head.{k}"]), "pt") as f:
                src[k] = f.get_tensor(f"lm_head.{k}")
    tr = src["trellis"]
    lin = LinearEXL3(None, tr.shape[0] * 16, tr.shape[1] * 16, None, None, None, src["suh"].to(dev), src["svh"].to(dev),
                     tr.to(dev), src.get("mcg"), src.get("mul1"), None, None, key="lm_head")
    print(f"source head K={lin.K} in {lin.in_features} out {lin.out_features}", flush=True)
    w = lin.get_weight_tensor()
    del lin, tr, src
    H_data = {"H": d["H"].to(dev), "first_key": "lm_head", "count": d["count"], "finalized": False,
              "num_total": d["count"] * w.shape[0], "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qa = {"seed": 0, "K": args.bits, "devices": [0], "apply_out_scales": True, "mul1": True,
          "debug_dir": os.path.join(args.work, "debug")}
    t0 = time.time()
    wf = w.cpu().float(); del w; torch.cuda.empty_cache()   # quantize_exl3 moves it: no second GPU copy held here
    _, proxy_err, out = qz.quantize_exl3(wf, H_data, qa, False, "lm_head <step>", verbose=True)
    print(f"proxy_err {proxy_err:.6f}  {time.time()-t0:.0f}s", flush=True)
    t = {f"lm_head.{k}": v.contiguous().cpu() for k, v in out.items()}
    for k, v in t.items(): print(k, tuple(v.shape), v.dtype, v.numel() * v.element_size() / 1e6, "MB")
    save_file(t, args.out)

if __name__ == "__main__":
    mode = sys.argv.pop(1)
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--work", required=True)
    if mode == "capture":
        from exllamav3 import model_init
        model_init.add_args(p, cache=True, add_draft_model_args=True)
        p.add_argument("--tokens", type=int, default=384)
        capture(p.parse_args())
    else:
        p.add_argument("-m", "--model_dir", required=True)
        p.add_argument("--bits", type=int, default=4); p.add_argument("--out", required=True)
        quant(p.parse_args())
