import torch, sys
t = torch.load(sys.argv[1]); M = int(sys.argv[2]); n_gu = int(sys.argv[3]) if len(sys.argv) > 3 else 544
t0 = t[:,0].min(); st=(t[:,0]-t0)/100; rd=(t[:,1]-t0)/100; en=(t[:,2]-t0)/100; wg=t[:,3].long()
first = {}
for i in range(len(t)):
    w=int(wg[i]); first[w]=min(first.get(w,1e9), float(st[i]))
fs = torch.tensor([first[w] for w in sorted(first)])
print("first start per wg: wg<144 max %.1f, wg>=144 min %.1f max %.1f" % (fs[:144].max(), fs[144:].min(), fs[144:].max()))
# gate/up body by start bucket
g = slice(M, M+n_gu)
b = (en[g]-rd[g])
for lo in range(0, 160, 20):
    m = (st[g]>=lo)&(st[g]<lo+20)
    if m.any(): print(f"gu start {lo:3d}-{lo+20:3d}: n {int(m.sum()):3d} body mean {b[m].mean():6.1f} max {b[m].max():6.1f}")
# per-wg: body of first unit vs wg id ranges
for lo in range(0, 288, 48):
    m = (wg[g]>=lo)&(wg[g]<lo+48)
    print(f"wg {lo:3d}-{lo+48}: gu units {int(m.sum())} body mean {b[m].mean():6.1f}")
# concurrency over time
for tt in range(0, int(en.max()), 20):
    act = ((st<=tt)&(en>tt)).sum()
    print(f"t {tt:4d}: active units {int(act)}")
