"""验证 bull_bear_paper.tex 的结构和环境括号配对。"""
from pathlib import Path
import re

p = Path("bull_bear_paper.tex")
src = p.read_text(encoding="utf-8")

# 用 raw 字符串
begin_re = re.compile(r"\\begin\{([^}]+)\}")
end_re   = re.compile(r"\\end\{([^}]+)\}")
begins = begin_re.findall(src)
ends   = end_re.findall(src)

print("=== environment balance ===")
print(f"  \\begin{{...}} count: {len(begins)}")
print(f"  \\end{{...}}   count: {len(ends)}")
print(f"  matched ?        : {len(begins) == len(ends)}")

# per-environment counts
from collections import Counter
b_cnt = Counter(begins)
e_cnt = Counter(ends)
unbalanced = [(k, b_cnt[k], e_cnt[k])
               for k in set(b_cnt) | set(e_cnt)
               if b_cnt[k] != e_cnt[k]]
if unbalanced:
    print("\n  unbalanced environments:")
    for k, b, e in sorted(unbalanced):
        print(f"    {k:20s}  begin={b}  end={e}")
else:
    print("  all environments balanced.")

print("\n=== section structure ===")
for i, ln in enumerate(src.split("\n"), start=1):
    if ln.lstrip().startswith(("\\section", "\\subsection")):
        snippet = ln.strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        print(f"  line {i:>5d}: {snippet}")

print(f"\n=== total lines: {len(src.splitlines())} ===")
