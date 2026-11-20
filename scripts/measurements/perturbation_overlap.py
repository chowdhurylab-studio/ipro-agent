import random, sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from ipro_agent.structure import Structure
from ipro_agent.perturb import choose_segment

st = Structure.load(str(_REPO / "examples" / "ompf_pore" / "2omf_A.pdb"))
res = list(st.residues("A"))
nums = [int(r.id[1]) for r in res]
pros = [int(r.id[1]) for r in res if r.get_resname().strip() == "PRO"]
print("n residues", len(res), "range", min(nums), max(nums), "n PRO", len(pros))

DP = [18,20,38,40,42,62,64,82,83,109,110,113,114,115,117,118,119,120,121,123,124,125,130,131,132]
dpset = set(DP)
print("n DP", len(DP))

rng = random.Random(12345)
N = 20000
hit = 0
lenhist = {}
for _ in range(N):
    seg = choose_segment(st, "A", (1,5), rng)
    sn = [int(r.id[1]) for r in seg]
    lenhist[len(sn)] = lenhist.get(len(sn),0)+1
    if dpset & set(sn):
        hit += 1
print("P(segment touches any DP) =", hit/N)
print("length histogram", dict(sorted(lenhist.items())))

# analytic: uniform over proline-free windows of length L, L uniform 1..5
tot_p = 0.0
for L in range(1,6):
    # enumerate all windows of length L with consecutive numbering and no PRO
    wins = []
    for i in range(len(res)-L+1):
        chunk = res[i:i+L]
        ns = [int(r.id[1]) for r in chunk]
        if any(r.get_resname().strip()=="PRO" for r in chunk): continue
        if ns != list(range(ns[0], ns[0]+L)): continue
        wins.append(ns)
    h = sum(1 for w in wins if dpset & set(w))
    print(f"  L={L}: windows={len(wins)} touching={h} P={h/len(wins):.4f}")
    tot_p += (1/5)*(h/len(wins))
print("analytic P =", round(tot_p,4))
