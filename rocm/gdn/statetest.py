"""Exactness of the GDN log-mode rewind: for each accepted count a, the live state, conv state and logits after an
M=8 history forward + rewind must be bit-identical between legacy history (copy) and log mode (replay).
usage: python statetest.py -m MODEL -dm DRAFT -cs 16384"""
import sys, os, torch
def main():
    import argparse
    from exllamav3 import model_init
    from exllamav3.ext import exllamav3_ext as ext
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, add_draft_model_args=True)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    with torch.inference_mode():
        model, config, cache, tok = model_init.init(args)[:4]
        layers = list(cache.get_all_recurrent_layers().values())
        P = 600; M = 8
        torch.manual_seed(0)
        ids = torch.randint(100, 20000, (1, P + 16))
        base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 2048)}
        prm = dict(base, past_len=0)
        model.forward(input_ids=ids[:, :P], params=prm)
        rs = prm["recurrent_states"]
        s0 = [(l.recurrent_state[0, 0].clone(), l.conv_state[0].clone()) for l in layers]
        def run(mode, r):
            ext.gdn_replay_set(mode)
            for l, (a, c) in zip(layers, s0):
                l.recurrent_state[0, 0].copy_(a); l.conv_state[0].copy_(c)
            rs[0].position = P; rs[0].last_history = 0
            lg = model.forward(input_ids=ids[:, P:P + M], params=dict(base, past_len=P, recurrent_states=rs, recurrent_history=True))
            lm = layers[0].log_mode[0]
            rs[0].rewind(r)
            torch.cuda.synchronize()
            return lg.clone(), [(l.recurrent_state[0, 0].clone(), l.conv_state[0, :, :4].clone()) for l in layers], lm
        bad = 0
        for rep in range(args.reps):
            for r in range(M):
                la, sa, ma = run(0, r)
                lb, sb, mb = run(1, r)
                st = sum(not torch.equal(x[0], y[0]) for x, y in zip(sa, sb))
                cv = sum(not torch.equal(x[1], y[1]) for x, y in zip(sa, sb))
                md = max((x[0] - y[0]).abs().max().item() for x, y in zip(sa, sb))
                lgq = torch.equal(la, lb)
                ok = st == 0 and cv == 0 and lgq and ma is False and mb is True
                bad += not ok
                print(f"rep {rep} rejected {r}: log_mode {ma}/{mb} state layers differing {st}/{len(layers)} (max abs {md:.3g}) conv {cv} logits equal {lgq} {'OK' if ok else 'FAIL'}", flush=True)
        print("ALL EXACT" if bad == 0 else f"FAILURES {bad}")
if __name__ == "__main__":
    main()
