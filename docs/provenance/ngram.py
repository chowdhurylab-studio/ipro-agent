import re, sys

def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [w for w in s.split() if w]

paper = norm(open(sys.argv[1], encoding="utf-8", errors="replace").read())
pset = set()
N = 8
for i in range(len(paper) - N + 1):
    pset.add(tuple(paper[i:i+N]))

quote = norm(open(sys.argv[2], encoding="utf-8", errors="replace").read())
# mark which N-gram starts match
hit = [tuple(quote[i:i+N]) in pset for i in range(len(quote)-N+1)]
# find maximal runs of consecutive matching n-grams -> verbatim word runs
runs = []
i = 0
while i < len(hit):
    if hit[i]:
        j = i
        while j+1 < len(hit) and hit[j+1]:
            j += 1
        runs.append((i, j+N))  # word indices [i, j+N)
        i = j+1
    else:
        i += 1
runs.sort(key=lambda r: r[0]-r[1])
for a,b in runs:
    print(f"[{b-a} words] {' '.join(quote[a:b])}")
print(f"\ntotal quote words: {len(quote)}")
print(f"verbatim words covered: {sum(b-a for a,b in runs)}")
