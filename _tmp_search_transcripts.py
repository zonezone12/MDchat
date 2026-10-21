import json
import re
from pathlib import Path

root = Path(r"C:\Users\zonezone\.cursor\projects\c-Users-zonezone-github-MD-analysis\agent-transcripts")
out = Path(r"c:\Users\zonezone\github\MD_analysis\_tmp_search_out.txt")

# Standalone 2' / 2-prime / 2′, not possessives like G2's or quoted ids like 'l2'
token = re.compile(
    r"(?<![A-Za-z0-9_])2['\u2032\u2035](?![A-Za-z])"
    r"|2\s*[- ]\s*prime"
    r"|2\u2086"
    r"|2₆",
    re.I,
)

lines_out = []
files = sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
lines_out.append(f"parent files {len(files)}")

for p in files:
    text = p.read_text(encoding="utf-8", errors="replace")
    user_qs = []
    for line in text.splitlines():
        if '"role":"user"' not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("role") != "user":
            continue
        content = obj.get("message", {}).get("content", "")
        if isinstance(content, list):
            bits = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    bits.append(c.get("text", ""))
            content = "\n".join(bits)
        elif not isinstance(content, str):
            content = str(content)
        m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", content, re.S)
        q = (m.group(1).strip() if m else content.strip())
        user_qs.append(q)

    # user-query hits
    user_hits = []
    for i, q in enumerate(user_qs, 1):
        found = token.findall(q)
        if found:
            user_hits.append((i, found, q[:500].replace("\n", " ")))

    # body hits excluding user wrapper noise: count occurrences
    body_n = len(token.findall(text))
    if not user_hits and body_n == 0:
        continue

    mtime = p.stat().st_mtime
    lines_out.append("=" * 80)
    lines_out.append(f"ID {p.parent.name} mtime {mtime:.0f} body_hits {body_n} user_msgs {len(user_qs)}")
    if user_qs:
        lines_out.append("TITLE_SRC " + user_qs[0][:180].replace("\n", " "))
    if user_hits:
        lines_out.append(f"USER_HITS {len(user_hits)}")
        for i, found, q in user_hits[:8]:
            lines_out.append(f"  Q{i} tokens={found}")
            lines_out.append(f"    {q}")
    else:
        # sample a few body contexts
        shown = 0
        for m in token.finditer(text):
            a = max(0, m.start() - 60)
            b = min(len(text), m.end() + 60)
            snip = text[a:b].replace("\n", " ")
            lines_out.append("  BODY " + snip)
            shown += 1
            if shown >= 3:
                break

out.write_text("\n".join(lines_out), encoding="utf-8")
print(f"wrote {out} lines {len(lines_out)}")
