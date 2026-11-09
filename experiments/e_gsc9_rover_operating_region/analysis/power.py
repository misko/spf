import pickle, numpy as np, collections
S='/tmp/claude-1000/-home-mouse9911-gits-spf/fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/'
H=pickle.load(open(S+'gsc9/rover_hist.pkl','rb'))
G=set(range(23,63)); REF=62
print("=== 1. coverage of the [23,62]^2 grid, END-of-buffer cells ===")
for lo in (5766000000,5840000000):
    for tag in ('all','clean'):
        c,n=H[f"{tag}_{lo}"]
        inside=sum(v for k,v in c.items() if k[0] in G and k[1] in G)
        outc=[k for k in c if not(k[0] in G and k[1] in G)]
        print(f"  {tag} {lo/1e6:.0f}: frames {100*inside/n:.4f}%  cells inside {sum(1 for k in c if k[0] in G and k[1] in G)}/{len(c)}  outside cells={sorted(outc)[:8]}")
# start-cell union from the raw scan
a=np.load(S+'ladder2/scan_rows.npy',allow_pickle=True)
print("\n=== 2. coverage including START-of-buffer cells (union) ===")
for LO in (5766000000,5840000000):
    tot=0; ins=0; out=collections.Counter()
    for d in a:
        lo=np.asarray(d['rx_lo']).astype(np.int64); m=lo==LO
        if not m.any(): continue
        ge=d['gains'][m].astype(int); gs=np.asarray(d['gain_db_start'])[m].astype(int)
        for arr in (ge,gs):
            for r in arr:
                tot+=1
                if r[0] in G and r[1] in G: ins+=1
                else: out[(r[0],r[1])]+=1
    print(f"  {LO/1e6:.0f}: {100*ins/tot:.4f}% of start+end cells inside; worst outside: {out.most_common(5)}")
print("\n=== 3. rover-mass split: directly in the additive CROSS (g1=62 or g2=62) vs HELD-OUT ===")
for lo in (5766000000,5840000000):
    c,n=H[f"all_{lo}"]
    cross=sum(v for k,v in c.items() if k[0]==REF or k[1]==REF)
    ho=sum(v for k,v in c.items() if k[0]!=REF and k[1]!=REF and k[0] in G and k[1] in G)
    w=np.array([v for k,v in c.items() if k[0]!=REF and k[1]!=REF and k[0] in G and k[1] in G],dtype=float)
    neff=w.sum()**2/ (w**2).sum()
    print(f"  {lo/1e6:.0f}: cross {100*cross/n:.2f}%  held-out {100*ho/n:.2f}% over {len(w)} cells, N_eff={neff:.1f}")
print("\n=== 4. power ===")
for sc in (0.3,0.5,0.7):
    for n in (3,4,5):
        nu=sc*np.sqrt(3/n); N=1521
        tmin=np.sqrt(3*np.sqrt(2/N)*nu**2)
        print(f"  sigma_c={sc:.1f} n={n}: per-cell resid sd={nu:.3f} deg | per-cell 3-sigma det={3*nu:.2f} | Bonferroni(1521,0.05) det={4.28*nu:.2f} | aggregate tau 3-sigma det={tmin:.3f} deg | adjacent-diagonal step 3-sigma det={3*sc*np.sqrt(2/n):.2f} deg | anchor sep 1 sem={sc*np.sqrt(2/n):.3f} deg")
print("\n=== 5. predicted tone_dbfs on the grid, per (radio,LO,arm) K measured 2026-08-11..13, tx=-29, 0.98 dB/dB ===")
K={('R17',5766,1):-45.64,('R17',5766,2):-46.17,('R17',5840,1):-44.18,('R17',5840,2):-45.99,
   ('R18',5766,1):-47.14,('R18',5766,2):-44.68,('R18',5840,1):-49.51,('R18',5840,2):-44.76}
T=-29.0
for g in (23,26,35,41,45,46,49,52,56,62):
    v=[0.98*T+g+k for k in K.values()]
    print(f"   g={g:3d}: tone_dbfs min {min(v):7.2f}  max {max(v):7.2f}  (median {np.median(v):7.2f})")
print(f"   -> clip margin at g=62: {-max(0.98*T+62+k for k in K.values()):.2f} dB below 0 dBFS (amplitude-referred)")
