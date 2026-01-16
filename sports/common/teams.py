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
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken",
    "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
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
