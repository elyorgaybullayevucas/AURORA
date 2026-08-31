import re

tex = open("main.tex", encoding="utf-8").read()
bib = open("refs.bib", encoding="utf-8").read()

used = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        used.add(k.strip())

defined = set(re.findall(r"@\w+\{([^,]+),", bib))
print("used    :", len(used))
print("defined :", len(defined))
print("missing :", sorted(used - defined) or "none")
print("unused  :", sorted(defined - used) or "none")
