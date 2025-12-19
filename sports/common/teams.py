import re
import unicodedata

ALIASES = {
    "St. Louis Blues": "St Louis Blues",
    "Montréal Canadiens": "Montreal Canadiens",
}

def canon_team(name: str) -> str:
    if not name:
        return ""
    n = name.strip()
    n = ALIASES.get(n, n)
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")
    n = re.sub(r"[’'`\.]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n
