import sys,re,collections
for f in sys.argv[1:]:
    L=open(f).read().split('\n'); lab={}
    for i,l in enumerate(L):
        m=re.match(r'^(\.LBB\d+_\d+):',l)
        if m: lab[m[1]]=i
    best=None
    for i,l in enumerate(L):
        m=re.search(r's_cbranch_\w+\s+(\.LBB\d+_\d+)',l) or re.search(r's_branch\s+(\.LBB\d+_\d+)',l)
        if m and m[1] in lab and lab[m[1]]<i:
            body=L[lab[m[1]]:i+1]; nd=sum('dot2' in x for x in body)
            if best is None or nd>best[0]: best=(nd,lab[m[1]],i,body)
    nd,s,e,body=best
    c=collections.Counter(x.split()[0] for x in body if x.strip() and not x.strip().startswith((';','.')))
    valu=sum(v for k,v in c.items() if k.startswith('v_')); dual=sum(v for k,v in c.items() if k.startswith('v_dual'))
    print(f"{f}: loop {s}-{e} lines {e-s}, dot2-lines {nd}, valu-lines {valu} (v_dual {dual}), s_waitcnt {c['s_waitcnt']}, s_nop {c['s_nop']}, s_delay_alu {c['s_delay_alu']}")
    print("   ", ", ".join(f"{k}:{v}" for k,v in c.most_common(22)))
