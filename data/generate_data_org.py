"""
Generate a synthetic B2B organization CSV with known duplicates and seeded hard cases.

Writes two files:
  <output>.csv        records to match (no ground truth inside, so nothing can leak)
  <output>_truth.csv  record_id -> true_entity_id, variant tags, decoy group

Seeded hard cases:
  acronym        "Summit Ridge Logistics Inc" -> "SRL Inc" / "S.R.L."
  truncation     name cut at a fixed field width, often mid-word
  po_box         street address replaced by a PO Box (phone often missing)
  decoy          two DIFFERENT entities with the same name in different cities

Standard noise: typos, case changes, legal-suffix forms, "&"/"and", street
abbreviations, phone formats, ZIP+4, missing fields.

Pure standard library: no Faker, fully reproducible from --seed.
"""

from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

DEFAULT_ROWS = 10_000
DEFAULT_SEED = 2026          # different from the seed used during tuning (42)
DECOY_SHARE = 0.05           # share of entities that get a same-name twin
P_ACRONYM = 0.12             # per duplicate record, if the name is eligible
P_TRUNCATION = 0.12
P_PO_BOX = 0.10
SOURCES = ["CRM", "ERP", "Billing", "Marketing"]
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "orgs_raw.csv"

CORES = [
    "Summit", "Riverside", "Granite", "Lakeshore", "Pinecrest", "Harbor", "Northgate",
    "Bluewater", "Ironwood", "Silverline", "Redstone", "Oakmont", "Westfield",
    "Brightpath", "Cedar", "Meridian", "Highland", "Clearview", "Stonebridge",
    "Evergreen", "Crescent", "Pacific", "Atlantic", "Frontier", "Liberty", "Keystone",
    "Heritage", "Sterling", "Prairie", "Coastal", "Ridge", "Valley", "Falcon",
    "Beacon", "Pioneer", "Copper", "Maple", "Willow", "Eastwood", "Sunrise",
]
SURNAMES = [
    "Anderson", "Patel", "Nguyen", "Garcia", "Thompson", "Brooks", "Hayes", "Morgan",
    "Reed", "Carter", "Foster", "Bennett", "Kim", "Shah", "Rivera", "Collins", "Ward",
    "Price", "Hughes", "Barnes", "Okafor", "Lindqvist", "Moreau", "Tanaka",
]
INDUSTRIES = [
    "Logistics", "Health Partners", "Financial Services", "Engineering Group",
    "Medical Supply", "Data Systems", "Construction", "Insurance Agency", "Dental Group",
    "Capital Management", "Manufacturing", "Technologies", "Consulting Group",
    "Energy Solutions", "Packaging", "Foods", "Freight Lines", "Pharmacy", "Realty",
    "Software", "Environmental Services", "Staffing Solutions", "Equipment Rental",
]
# canonical suffix -> variants seen in the wild
SUFFIXES = {
    "Inc": ["Inc", "Inc.", "Incorporated", "INC"],
    "LLC": ["LLC", "L.L.C.", "Llc"],
    "Corp": ["Corp", "Corp.", "Corporation"],
    "Co": ["Co", "Co.", "Company"],
    "Ltd": ["Ltd", "Ltd.", "Limited"],
    "": [""],
}
STOPWORDS = {"&", "and", "of", "the"}
CITIES = [
    ("Houston", "TX", "770"), ("Dallas", "TX", "752"), ("Austin", "TX", "787"),
    ("Chicago", "IL", "606"), ("Denver", "CO", "802"), ("Phoenix", "AZ", "850"),
    ("Atlanta", "GA", "303"), ("Seattle", "WA", "981"), ("Portland", "OR", "972"),
    ("Boston", "MA", "021"), ("Columbus", "OH", "432"), ("Charlotte", "NC", "282"),
    ("Nashville", "TN", "372"), ("Minneapolis", "MN", "554"), ("Tampa", "FL", "336"),
    ("Kansas City", "MO", "641"), ("Omaha", "NE", "681"), ("Raleigh", "NC", "276"),
    ("Salt Lake City", "UT", "841"), ("Pittsburgh", "PA", "152"),
    ("Sacramento", "CA", "958"), ("San Diego", "CA", "921"), ("Detroit", "MI", "482"),
    ("Milwaukee", "WI", "532"), ("Richmond", "VA", "232"), ("Albany", "NY", "122"),
]
STREET_NAMES = [
    "Main", "Oak", "Commerce", "Industrial", "Park", "Lake", "Market", "Center",
    "Washington", "Lincoln", "Jefferson", "Mill", "Highland", "Sunset", "Ridge",
    "Elm", "Cedar", "Pine", "Maple", "River", "Airport", "Corporate", "Enterprise",
]
STREET_TYPES = {
    "Street": "St", "Avenue": "Ave", "Boulevard": "Blvd", "Drive": "Dr",
    "Road": "Rd", "Parkway": "Pkwy", "Lane": "Ln", "Court": "Ct", "Way": "Way",
}


# --------------------------------------------------------------------------- canonical

def make_name(rng: random.Random) -> tuple[str, str]:
    """Return (base name without suffix, canonical suffix key)."""
    roll = rng.random()
    if roll < 0.45:
        base = f"{rng.choice(CORES)} {rng.choice(INDUSTRIES)}"
    elif roll < 0.75:
        c1, c2 = rng.sample(CORES, 2)
        base = f"{c1} {c2} {rng.choice(INDUSTRIES)}"
    elif roll < 0.9:
        s1, s2 = rng.sample(SURNAMES, 2)
        base = f"{s1} & {s2} {rng.choice(INDUSTRIES)}"
    else:
        base = f"{rng.choice(SURNAMES)} {rng.choice(INDUSTRIES)}"
    return base, rng.choice(list(SUFFIXES))


def make_location(rng: random.Random) -> dict:
    city, state, zip3 = rng.choice(CITIES)
    st_long = rng.choice(list(STREET_TYPES))
    return {
        "street": f"{rng.randint(10, 9999)} {rng.choice(STREET_NAMES)} {st_long}",
        "city": city,
        "state": state,
        "zip": f"{zip3}{rng.randint(0, 99):02d}",
    }


def make_phone(rng: random.Random) -> str:
    return f"{rng.randint(201, 989)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"


def domain_for(base: str) -> str:
    words = [w.lower() for w in base.replace("&", " ").split() if w.lower() not in STOPWORDS]
    return "".join(words[:2]) + ".com"


def canonical_entity(rng: random.Random, used: set[str]) -> dict:
    for _ in range(1000):
        base, suffix = make_name(rng)
        if base not in used:
            used.add(base)
            break
    else:
        raise RuntimeError("Ran out of unique names; add words to the pools")
    return {"base": base, "suffix": suffix, **make_location(rng),
            "phone": make_phone(rng), "website": domain_for(base)}


# --------------------------------------------------------------------------- noise

def typo(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s
    chars = list(s)
    i = rng.randrange(1, len(chars) - 1)
    kind = rng.choice(["swap", "drop", "dup", "sub"])
    if kind == "swap":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif kind == "drop":
        chars.pop(i)
    elif kind == "dup":
        chars.insert(i, chars[i])
    elif chars[i].isalpha():
        chars[i] = rng.choice(string.ascii_lowercase)
    return "".join(chars)


def acronym_of(base: str) -> str | None:
    words = [w for w in base.split() if w.lower() not in STOPWORDS]
    return "".join(w[0].upper() for w in words) if len(words) >= 3 else None


def full_name(base: str, suffix_key: str, rng: random.Random, vary: bool) -> str:
    suffix = rng.choice(SUFFIXES[suffix_key]) if vary else suffix_key
    if vary and "&" in base and rng.random() < 0.4:
        base = base.replace("&", "and")
    name = f"{base} {suffix}".strip()
    if vary and suffix and rng.random() < 0.3:
        name = name.replace(f" {suffix}", f", {suffix}")
    return name


def format_phone(digits: str, rng: random.Random) -> str:
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return rng.choice([f"{a}-{b}-{c}", f"({a}) {b}-{c}", f"{a}.{b}.{c}", digits, f"+1 {a} {b} {c}"])


def format_street(street: str, rng: random.Random) -> str:
    for long, short in STREET_TYPES.items():
        if street.endswith(long) and rng.random() < 0.5:
            street = street[: -len(long)] + short + rng.choice(["", "."])
    if rng.random() < 0.1:
        street += rng.choice([" Suite 100", " Ste 210", " #4", " Fl 2"])
    return street


def po_box(rng: random.Random) -> str:
    n = rng.randint(100, 99999)
    return rng.choice([f"PO Box {n}", f"P.O. Box {n}", f"POB {n}", f"Post Office Box {n}"])


def make_record(ent: dict, rng: random.Random, clean: bool) -> tuple[dict, list[str]]:
    tags: list[str] = []
    vary = not clean
    name = full_name(ent["base"], ent["suffix"], rng, vary)
    street = format_street(ent["street"], rng) if vary else ent["street"]
    phone = format_phone(ent["phone"], rng)
    zip_code = ent["zip"]
    website = ent["website"]

    if vary:
        acr = acronym_of(ent["base"])
        roll = rng.random()
        if acr and roll < P_ACRONYM:
            suffix = rng.choice(SUFFIXES[ent["suffix"]])
            name = rng.choice([f"{acr} {suffix}", ".".join(acr) + ". " + suffix, acr]).strip()
            tags.append("acronym")
        elif roll < P_ACRONYM + P_TRUNCATION:
            width = rng.randint(16, 24)
            if len(name) > width:
                name = name[:width].rstrip(" ,.")
                tags.append("truncation")
        elif roll < P_ACRONYM + P_TRUNCATION + P_PO_BOX:
            street = po_box(rng)
            if rng.random() < 0.4:
                phone = ""
            tags.append("po_box")

        if rng.random() < 0.10:
            name = typo(name, rng)
            tags.append("typo")
        if rng.random() < 0.10:
            name = name.upper()
            tags.append("case")
        if rng.random() < 0.06:
            street = typo(street, rng)
            tags.append("typo_address")
        if rng.random() < 0.20:
            zip_code = f"{zip_code}-{rng.randint(0, 9999):04d}"
        if rng.random() < 0.10 and phone:
            phone = ""
            tags.append("missing_phone")
        if rng.random() < 0.30:
            website = ""
        elif rng.random() < 0.2:
            website = "www." + website

    record = {
        "company_name": name,
        "address_line": street,
        "city": ent["city"].upper() if vary and rng.random() < 0.1 else ent["city"],
        "state": ent["state"],
        "postal_code": zip_code,
        "phone": phone,
        "website": website,
        "source_system": rng.choice(SOURCES),
    }
    return record, tags or ["clean" if clean else "standard"]


# --------------------------------------------------------------------------- build

def records_per_entity(rng: random.Random) -> int:
    return rng.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[35, 25, 15, 10, 6, 4, 3, 2])[0]


def build(rows: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    used: set[str] = set()
    entities: list[dict] = []
    counts: list[int] = []
    total = 0
    while total < rows:
        ent = canonical_entity(rng, used)
        ent["decoy_group"] = ""
        k = min(records_per_entity(rng), rows - total)
        entities.append(ent)
        counts.append(k)
        total += k

    # Decoys: give some entities a different twin entity with the same name.
    n_decoys = int(len(entities) * DECOY_SHARE)
    for g, idx in enumerate(rng.sample(range(len(entities)), n_decoys), start=1):
        src = entities[idx]
        twin = {"base": src["base"], "suffix": src["suffix"], **make_location(rng),
                "phone": make_phone(rng),
                "website": src["website"] if rng.random() < 0.5 else "",
                "decoy_group": f"D{g:04d}"}
        while twin["city"] == src["city"]:
            twin.update(make_location(rng))
        src["decoy_group"] = twin["decoy_group"]
        # take one row from the largest entity so the total stays fixed
        donor = max(range(len(counts)), key=counts.__getitem__)
        counts[donor] -= 1
        entities.append(twin)
        counts.append(1)

    records, truth = [], []
    for eid, (ent, k) in enumerate(zip(entities, counts), start=1):
        for j in range(k):
            rec, tags = make_record(ent, rng, clean=(j == 0))
            rid = f"R{rng.getrandbits(40):010x}"
            records.append({"record_id": rid, **rec})
            truth.append({"record_id": rid, "true_entity_id": f"E{eid:05d}",
                          "variants": "+".join(tags), "decoy_group": ent["decoy_group"]})

    order = list(range(len(records)))
    rng.shuffle(order)
    return [records[i] for i in order], [truth[i] for i in order]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()

    records, truth = build(args.rows, args.seed)
    truth_path = args.output.with_name(args.output.stem + "_truth.csv")
    write_csv(args.output, records)
    write_csv(truth_path, truth)

    n_ent = len({t["true_entity_id"] for t in truth})
    tag_counts: dict[str, int] = {}
    for t in truth:
        for tag in t["variants"].split("+"):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"{len(records)} records, {n_ent} entities (seed {args.seed})")
    print(f"  records -> {args.output}")
    print(f"  truth   -> {truth_path}")
    for tag in sorted(tag_counts):
        print(f"  {tag:<14}{tag_counts[tag]:>6}")
    n_groups = len({t["decoy_group"] for t in truth if t["decoy_group"]})
    print(f"  decoy groups  {n_groups:>6}")


if __name__ == "__main__":
    main()
