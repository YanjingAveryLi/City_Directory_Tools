#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City directory social organization entry schema and category taxonomy.

1) ENTRY_SCHEMA: common fields per entry
2) CATEGORY_TAXONOMY: per-category sub-types / name patterns for entry prediction and classification
"""

from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# 1) Entry schema: common fields for an org entry in directories
# ---------------------------------------------------------------------------

ENTRY_SCHEMA: List[Dict[str, str]] = [
    {"field": "org_name", "label": "Organization name (incl. abbreviations/aliases)"},
    {"field": "category", "label": "Category (Church / Club / Fraternal / Veterans ...)"},
    {"field": "address", "label": "Address (meeting place / office / chapel / lodge address)"},
    {"field": "contact", "label": "Contact (phone, P.O. Box; common in older directories)"},
    {"field": "officers", "label": "Officers (President / Secretary / Treasurer / Chaplain etc.)"},
    {"field": "affiliation", "label": "Affiliation (grand lodge / diocese / national HQ; lodge No.)"},
    {"field": "activity", "label": "Activity (meeting night, service times, audience)"},
    {"field": "function_tags", "label": "Function tags (relief / mutual aid / education / religious / social etc.)"},
]

ENTRY_SCHEMA_FIELDS: List[str] = [e["field"] for e in ENTRY_SCHEMA]

# Only these count as organization_category in output. Anything else -> Miscellaneous or Uncategorized.
# Structure: Churches, Clubs, Hospitals, Libraries, Organizations (Civic, Labor, Benevolent and Fraternal,
# Patriotic and Veterans, Welfare and Relief, Young People, Miscellaneous, Secret Societies, etc.)
ALLOWED_ORGANIZATION_CATEGORIES: List[str] = [
    "Churches",
    "Clubs",
    "Hospitals",
    "Libraries",
    "Civic Organizations",
    "Labor Organizations",
    "Benevolent and Fraternal",
    "Patriotic Organizations",
    "Veterans Organizations",
    "Welfare Organizations",
    "Relief Organizations",
    "Youth Organizations",
    "Miscellaneous",
    "Secret Societies",
    "Uncategorized",
]


# ---------------------------------------------------------------------------
# 2) Category taxonomy: top-level category -> sub-types / entry hints for rules and LLM
# ---------------------------------------------------------------------------

CATEGORY_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "Churches": {
        "label": "Religious institutions",
        "entry_hint": "Church name + address + pastor/priest + service times",
        "sub_types": [
            "Mainline denominations: Catholic, Baptist, Methodist, Presbyterian, Lutheran, Episcopal, Congregational, Pentecostal",
            "Independent: Independent Bible Church, Gospel Hall",
            "Ethnic/language: Chinese Church, German Lutheran, African Methodist Episcopal",
            "Affiliated: women's society, youth fellowship, mission, charity (e.g. St. Vincent de Paul Society)",
        ],
        "name_patterns": ["church", "chapel", "parish", "cathedral", "mission", "synagogue", "temple", "catholic", "christian science"],
    },
    "Clubs": {
        "label": "Clubs and societies",
        "entry_hint": "By interest / class / activity",
        "sub_types": [
            "Social: City Club, Women's Club, Business Men's Club",
            "Sports/outdoor: Hunting & Fishing Club, Golf Club, Yacht Club, Athletic Club",
            "Literary/arts: Literary Club, Music Society, Art League, Debating Society",
            "Service: Rotary, Lions, Kiwanis, Optimist (sometimes civic)",
            "Ethnic: Irish Club, Italian Society, Greek Society",
            "Trade: Press Club, Real Estate Board (may be civic/commerce)",
        ],
        "name_patterns": ["club", "society club", "society", "athletic", "sport", "recreation", "reading", "literary", "music", "dramatic", "golf", "tennis", "yacht", "band", "art league"],
    },
    "Hospitals": {
        "label": "Hospitals and care",
        "entry_hint": "Name + address + officers/departments",
        "sub_types": [
            "General: General Hospital, City Hospital",
            "Specialty: Maternity Hospital, Children's Hospital, Eye & Ear Infirmary",
            "Historic: Sanatorium, Tuberculosis Hospital, Asylum",
            "Charity/faith: St. Mary's Hospital, Catholic Hospital",
            "Nursing: Nursing Home, Visiting Nurse Association (sometimes welfare/relief)",
        ],
        "name_patterns": ["hospital", "infirmary", "clinic", "sanitarium", "sanatorium", "nursing"],
    },
    "Libraries": {
        "label": "Libraries and reading rooms",
        "entry_hint": "Name + address",
        "sub_types": [
            "Public: Public Library, Carnegie Library (historic)",
            "College: College Library",
            "Association: Mercantile Library, Athenaeum, Reading Room",
            "Law/professional: Law Library (may be under court/bar)",
        ],
        "name_patterns": ["library", "reading room", "reading-room", "athenaeum"],
    },
    "Parks": {
        "label": "Parks and recreation",
        "entry_hint": "Park name + address/transit",
        "sub_types": ["Parks, gardens, playing grounds, recreation grounds"],
        "name_patterns": ["park", "garden", "recreation ground"],
    },
    "Theaters and Entertainment": {
        "label": "Theaters and entertainment",
        "entry_hint": "Venue name + address + manager/lessee",
        "sub_types": ["Opera House, Auditorium, Theater, Pavilion, Playhouse"],
        "name_patterns": ["opera house", "opera", "auditorium", "theater", "theatre", "pavilion", "playhouse"],
    },
    "Halls": {
        "label": "Halls",
        "entry_hint": "Hall name + address",
        "sub_types": ["Meeting hall, concert hall, ballroom, rental hall"],
        "name_patterns": ["hall"],
    },
    "Banks": {
        "label": "Banks and financial",
        "entry_hint": "Bank name + address (sometimes building name only)",
        "sub_types": ["Bank, National Bank, Savings Bank, Bank building"],
        "name_patterns": ["bank"],
    },
    "Transportation": {
        "label": "Transportation",
        "entry_hint": "Company name + address/agent",
        "sub_types": ["Express Company, Railway, Railroad, Transit"],
        "name_patterns": ["express co", "express company", "railway", "railroad", "transit"],
    },
    "Civic Organizations": {
        "label": "Civic organizations",
        "entry_hint": "Public interest / city governance / commerce",
        "sub_types": [
            "Chamber of Commerce, Board of Trade",
            "Civic League, City Improvement Association, Taxpayers' League",
            "Neighborhood Association, Community Council",
            "Firemen's Association, Police Benevolent Association (may be benevolent)",
        ],
        "name_patterns": ["civic", "community", "neighborhood", "citizens", "council", "chamber of commerce", "board of trade", "league", "committee", "improvement association", "improvement league", "city improvement"],
    },
    "Labor Organizations": {
        "label": "Labor and unions",
        "entry_hint": "Local No., Union Hall address, meeting nights",
        "sub_types": [
            "Carpenters Union Local No. X, Teamsters Local",
            "Labor Council, Trades and Labor Assembly",
            "Bar Association, Medical Society (gray area)",
            "Workingmen's Association",
        ],
        "name_patterns": ["labor", "union", "trade", "brotherhood of", "workers", "guild", "local no"],
    },
    "Benevolent Societies": {
        "label": "Benevolent and mutual aid",
        "entry_hint": "Relief / mutual aid / benefit",
        "sub_types": [
            "Mutual Benefit Association, Sick & Death Benefit Society",
            "Charity, relief, aid societies",
        ],
        "name_patterns": ["benevolent", "charity", "charitable", "aid", "relief", "mutual aid", "mutual", "benefit"],
    },
    "Fraternal Organizations": {
        "label": "Fraternal and lodges",
        "entry_hint": "Lodge/Chapter/Court/Camp + No. + address + officers (W.M., Secretary)",
        "sub_types": [
            "Freemasons (Grand Lodge / Lodge No.), Scottish Rite, Shriners",
            "IOOF (Odd Fellows), Knights of Pythias, Foresters, Eagles, Elks",
            "Knights of Columbus (Catholic)",
        ],
        "name_patterns": ["fraternal", "lodge", "masonic", "odd fellows", "knights of", "elks", "moose", "eagles", "foresters", "pythias", "masons", "grange", "chapter", "commandery"],
    },
    "Benevolent and Fraternal": {
        "label": "Benevolent and fraternal (combined)",
        "entry_hint": "Lodge/Chapter/mutual aid/benefit societies",
        "sub_types": [
            "Benevolent: Mutual Benefit, charity, relief, aid societies",
            "Fraternal: Freemasons, IOOF, Knights of Pythias, Elks, Eagles, Ancient Order of Hibernians, etc.",
        ],
        "name_patterns": [
            "benevolent", "charity", "charitable", "aid", "relief", "mutual aid", "mutual", "benefit",
            "fraternal", "lodge", "masonic", "odd fellows", "knights of", "elks", "moose", "eagles", "foresters", "pythias", "masons", "grange", "chapter", "commandery", "hibernians", "order of ",
            "mystic circle", "mystic cirele", "g.a.r", "grand army", "sons of veterans", " post, no", " camp, no", "catholic knights", "i.o.b.r", "b.p.o.e",
            "tribe", "encampment", "triple link", " link, no", "i.o.o.f", "grand lodge", "d. of r", "rebekah",
            " council, no", "a.a.o.n", "a. a. o. n", "potentate",
        ],
    },
    "Patriotic Organizations": {
        "label": "Patriotic and memorial",
        "entry_hint": "War/history memorial, Sons/Daughters of ...",
        "sub_types": [
            "Grand Army of the Republic (GAR), Sons/Daughters of ...",
            "Honor Guard Association",
        ],
        "name_patterns": ["patriotic", "daughters of", "sons of", "d.a.r", "s.a.r", "american revolution"],
    },
    "Veterans Organizations": {
        "label": "Veterans organizations",
        "entry_hint": "Post No., branch/war era, meeting place",
        "sub_types": [
            "American Legion Post No., VFW Post, Disabled American Veterans (DAV)",
            "Grand Army of the Republic (GAR, historic)",
        ],
        "name_patterns": ["veteran", "veterans", "legion", "g.a.r", "grand army", "vfw", "post"],
    },
    "Welfare Organizations": {
        "label": "Welfare organizations",
        "entry_hint": "Target (poor / disaster / children / sick), relief function",
        "sub_types": [
            "Charity Organization Society, Relief Association",
            "Children's Aid Society, Orphanage Board",
            "Visiting Nurse Association, Public Health League",
        ],
        "name_patterns": ["welfare", "social service", "settlement", "rescue"],
    },
    "Relief Organizations": {
        "label": "Relief organizations",
        "entry_hint": "Disaster relief / religious charity / poor relief",
        "sub_types": [
            "Red Cross Chapter",
            "Salvation Army, Catholic Charities, Church Relief Society",
        ],
        "name_patterns": ["relief", "red cross", "salvation army"],
    },
    "Youth Organizations": {
        "label": "Youth organizations",
        "entry_hint": "Council/Troop number, venue (school / church / community center)",
        "sub_types": [
            "YMCA, YWCA",
            "Boy Scouts, Girl Scouts (Troop/Council)",
            "4-H, Youth League",
            "Young People's Society, Christian Endeavor (historic)",
        ],
        "name_patterns": ["youth", "young people", "ymca", "ywca", "boys", "girls", "scout", "troop", "council"],
    },
    "Secret Societies": {
        "label": "Secret societies (ritual / membership)",
        "entry_hint": "Less public info; often name and address only",
        "sub_types": [
            "University secret societies (college towns); ritual lodge/order branches (overlap with Masons/IOOF)",
        ],
        "name_patterns": ["secret", "order", "lodge", "chapter", "society", "fraternity"],
    },
    "Miscellaneous": {
        "label": "Miscellaneous",
        "entry_hint": "Does not fit other categories",
        "sub_types": [
            "Historical Society",
            "Horticultural Society",
            "Parent-Teacher Association (PTA)",
            "Humane Society",
            "Engineering Society, Pharmaceutical Association (sometimes professional)",
        ],
        "name_patterns": [],
    },
    "Buildings": {
        "label": "Buildings (building name only, no org type)",
        "entry_hint": "e.g. X bldg., address-only building name",
        "sub_types": ["Commercial/office building names"],
        "name_patterns": ["bldg.", "bldg", "building"],
    },
    "Uncategorized": {
        "label": "Uncategorized",
        "entry_hint": "Not an organization or unclear",
        "sub_types": [],
        "name_patterns": [],
    },
}


def get_top_level_categories() -> List[str]:
    """Return all top-level categories (for classifier allowed list)."""
    return list(CATEGORY_TAXONOMY.keys())


# If org name contains any of these, treat as not a social organization -> Uncategorized (e.g. insurance, B&L, gov bodies, schools).
EXCLUDE_ORGANIZATION_PATTERNS: List[str] = [
    "insurance",
    "underwriters",
    "indemnity",
    "building & loan",
    "building and loan",
    "board of education",
    "music school",
    " school",  # "X School" (leading space to match phrase)
]


def is_excluded_organization(name: str) -> bool:
    """True if name indicates insurance / B&L / similar (not a social org)."""
    if not (name or "").strip():
        return False
    t = (name or "").lower()
    return any(p in t for p in EXCLUDE_ORGANIZATION_PATTERNS)


def normalize_to_allowed_category(cat: str) -> str:
    """Map category to an allowed value only. Non-allowed -> Uncategorized. Legacy labels -> standard (e.g. Fraternal Organizations -> Benevolent and Fraternal)."""
    if not (cat or "").strip():
        return "Uncategorized"
    c = (cat or "").strip()
    if c in ALLOWED_ORGANIZATION_CATEGORIES:
        return c
    if c in ("Fraternal Organizations", "Benevolent Societies"):
        return "Benevolent and Fraternal"
    return "Uncategorized"


def get_name_patterns_for_rule_based() -> List[tuple]:
    """
    Return (category_label, [keywords]) for rule-based classification.
    Order matters: more specific patterns first (e.g. Fraternal before Churches for Masonic Temple).
    """
    order = [
        "Hospitals",
        "Libraries",
        "Benevolent and Fraternal",
        "Churches",
        "Parks",
        "Theaters and Entertainment",
        "Halls",
        "Banks",
        "Transportation",
        "Clubs",
        "Civic Organizations",
        "Labor Organizations",
        "Patriotic Organizations",
        "Veterans Organizations",
        "Welfare Organizations",
        "Relief Organizations",
        "Youth Organizations",
        "Secret Societies",
        "Buildings",
    ]
    out = []
    for cat in order:
        if cat not in CATEGORY_TAXONOMY:
            continue
        patterns = CATEGORY_TAXONOMY[cat].get("name_patterns") or []
        if patterns:
            out.append((cat, patterns))
    return out


def get_llm_category_descriptions() -> List[str]:
    """Return allowed categories only + short descriptions for LLM prompt. Only these are valid."""
    lines = []
    for cat in ALLOWED_ORGANIZATION_CATEGORIES:
        if cat in ("Uncategorized", "Miscellaneous"):
            continue
        info = CATEGORY_TAXONOMY.get(cat)
        if not info:
            continue
        label = info.get("label", "")
        hint = info.get("entry_hint", "")
        sub = info.get("sub_types", [])
        desc = f"- {cat}"
        if label:
            desc += f" ({label})"
        if hint:
            desc += f": {hint}"
        if sub and isinstance(sub[0], str) and len(sub[0]) < 80:
            desc += f" e.g. {sub[0][:60]}..." if len(sub[0]) > 60 else f" e.g. {sub[0]}"
        lines.append(desc)
    lines.append("- Miscellaneous (only when nothing else fits)")
    lines.append("- Uncategorized (not a social organization or unclear)")
    return lines
