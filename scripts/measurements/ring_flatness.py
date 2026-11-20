import sys, numpy as np
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from ipro_agent.pore import ring_flatness, axis_from_lining, vdw_radii
from ipro_agent.structure import Structure

pdb = str(_REPO / "examples" / "ompf_pore" / "2omf_A.pdb")
st = Structure.load(pdb)
full = [18,20,38,40,42,62,64,82,83,109,110,113,114,115,117,118,119,120,121,123,124,125,130,131,132]
smoke = [18,20,38,40]

def cas(nums):
    out=[]
    for n in nums:
        try: out.append(st.find("A", n).child_dict["CA"].coord)
        except Exception as e: print("missing", n, e)
    return np.array(out, dtype=float)

for name, nums in [("25-position full", full), ("4-position smoke", smoke), ("3 positions", smoke[:3])]:
    c = cas(nums)
    u,s,vt = np.linalg.svd(c - c.mean(axis=0), full_matrices=False)
    print(f"{name}: n={len(c)} singvals={np.round(s,3)} flatness={ring_flatness(c):.6g}")

# random clouds
rng = np.random.default_rng(0)
vals=[ring_flatness(rng.normal(size=(25,3))) for _ in range(8)]
print("25 random gaussian flatness:", np.round(vals,3))

# clearance for each spec: distance from fitted origin to nearest vdW surface (all protein atoms)
atoms = st.all_atoms()
coords = np.array([a.coord for a in atoms]); names=[a.get_name() for a in atoms]
r = vdw_radii(names, None)
for name, nums in [("25-position full", full), ("4-position smoke", smoke)]:
    c = cas(nums); o,d = axis_from_lining(c)
    clear = float(np.min(np.sqrt(((coords-o)**2).sum(-1)) - r))
    print(f"{name}: origin={np.round(o,2)} clearance(all atoms)={clear:.3f}")
