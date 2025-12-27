import re
import unicodedata

ALIASES = {
    "St. Louis Blues": "St Louis Blues",
    "Montréal Canadiens": "Montreal Canadiens",
}
# sports/common/teams.py
import re
import unicodedata

NHL_ALIASES = {
    "LA Kings": "Los Angeles Kings",
    "Los Angeles": "Los Angeles Kings",
    "NY Rangers": "New York Rangers",
    "NY Islanders": "New York Islanders",
    "NJ Devils": "New Jersey Devils",
    "Tampa Bay": "Tampa Bay Lightning",
    "Vegas": "Vegas Golden Knights",
    "St Louis": "St Louis Blues",
    "St. Louis": "St Louis Blues",
    "Montreal": "Montreal Canadiens",
    "Montréal": "Montreal Canadiens",
}

def canon_team(name: str) -> str:
    if not name:
        return ""

    n = name.strip()

    # normalize unicode
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode("ascii")

    # remove punctuation
    n = re.sub(r"[’'`\.]", "", n)

    # normalize whitespace
    n = re.sub(r"\s+", " ", n).strip()

    # alias resolution
    n = NHL_ALIASES.get(n, n)

    return n

