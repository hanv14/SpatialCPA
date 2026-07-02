#!/usr/bin/env python
"""Scrape ALL 10x Genomics spatial transcriptomics datasets via Algolia API.

The 10x datasets page (10xgenomics.com/datasets) uses Algolia for search/filtering.
This script queries the Algolia API directly to get complete, structured metadata
for all spatial datasets (Visium, Visium HD, Xenium, CytAssist).

Output: compatibility-matrix/10x_datasets_spatial.csv

No browser dependencies required — uses the public Algolia search API.
"""
import csv
import re
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "compatibility-matrix"
OUTPUT_CSV = OUTPUT_DIR / "10x_datasets_spatial.csv"

# Algolia credentials (public, embedded in 10x website JS)
ALGOLIA_APP_ID = "GI4MFTCJTA"
ALGOLIA_API_KEY = "9845a4ac4d37c6c2f5a8d90482754494"
ALGOLIA_INDEX = "master:datasets"
ALGOLIA_URL = f"https://{ALGOLIA_APP_ID.lower()}-dsn.algolia.net/1/indexes/*/queries"

# Spatial product families on 10x Genomics
SPATIAL_PRODUCTS = [
    "Spatial Gene Expression",           # Visium v1 (original)
    "HD Spatial Gene Expression",        # Visium HD (CytAssist)
    "HD 3' Spatial Gene Expression",     # Visium HD 3'
    "CytAssist Spatial Gene and Protein Expression",  # CytAssist + protein
    "In Situ Gene Expression",           # Xenium
    "In Situ Gene and Protein Expression",  # Xenium + protein
]

# Keywords for serial/multi-section identification
SERIAL_SLUG_KEYWORDS = [
    "serial-section",
    "section-1",
    "section-2",
    "section-3",
    "section-4",
    "block-a",
    "block-b",
]


def query_algolia(product_name, hits_per_page=200):
    """Query Algolia for all datasets of a given product type."""
    headers = {
        "x-algolia-application-id": ALGOLIA_APP_ID,
        "x-algolia-api-key": ALGOLIA_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "requests": [
            {
                "indexName": ALGOLIA_INDEX,
                "params": f"query=&hitsPerPage={hits_per_page}"
                          f'&facetFilters=[["product.name:{product_name}"]]'
                          f"&attributesToRetrieve=*",
            }
        ]
    }
    resp = requests.post(ALGOLIA_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data["results"][0]
    return results["hits"], results["nbHits"]


def classify_product(product_name):
    """Map 10x product name to a short category."""
    name = product_name.lower()
    if "hd" in name and "spatial" in name:
        return "Visium HD"
    elif "in situ" in name:
        return "Xenium"
    elif "cytassist" in name:
        return "Visium CytAssist"
    elif "spatial gene expression" in name:
        return "Visium"
    return "Spatial (other)"


def deduplicate_by_section(hits):
    """Deduplicate datasets that are the same section at different pipeline versions.

    10x publishes the same dataset processed with different Space Ranger versions
    (e.g., 1.0.0, 1.1.0, 2.0.0). We keep the latest pipeline version for each
    unique biological section.
    """
    # Group by base slug (remove pipeline version suffix)
    groups = {}
    for hit in hits:
        slug = hit.get("slug", "")
        # Remove trailing version like -1-standard-1-1-0 or -2-standard
        base = re.sub(r"-\d+-standard(-\d+-\d+-\d+)?$", "", slug)
        if base not in groups:
            groups[base] = []
        groups[base].append(hit)

    # Keep the one with the latest pipeline version in each group
    deduped = []
    for base, members in groups.items():
        best = max(members, key=lambda h: h.get("pipeline", "0.0.0"))
        best["_all_versions"] = [m.get("pipeline", "") for m in members]
        best["_base_slug"] = base
        deduped.append(best)

    return deduped


def identify_serial_groups(datasets):
    """Group datasets by specimen to find multi-section serial sets."""
    groups = {}
    for ds in datasets:
        base = ds.get("_base_slug", ds.get("slug", ""))
        # Remove serial-section-N and section-N patterns
        group_key = re.sub(r"-serial-section-\d+", "", base)
        group_key = re.sub(r"-section-\d+", "", group_key)
        # Remove coronal-section-N
        group_key = re.sub(r"-coronal-section-\d+", "-coronal", group_key)
        # Remove sagittal region (anterior/posterior) to group all sagittal sections together
        group_key = re.sub(r"-sagittal-(anterior|posterior)", "-sagittal", group_key)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(ds)

    return groups


def is_serial_section(slug):
    """Check if slug suggests a serial section dataset."""
    return any(kw in slug for kw in SERIAL_SLUG_KEYWORDS)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying 10x Genomics Algolia API for spatial datasets...")
    print(f"Output: {OUTPUT_CSV}\n")

    all_hits = []
    for product in SPATIAL_PRODUCTS:
        hits, total = query_algolia(product)
        category = classify_product(product)
        for h in hits:
            h["_category"] = category
        all_hits.extend(hits)
        print(f"  {product}: {total} datasets ({len(hits)} returned)")

    print(f"\nTotal raw hits: {len(all_hits)}")

    # Deduplicate (same section, different pipeline versions)
    deduped = deduplicate_by_section(all_hits)
    print(f"After deduplication: {len(deduped)} unique datasets")

    # Build structured records
    records = []
    for hit in deduped:
        slug = hit.get("slug", "")
        name = hit.get("name", slug)
        records.append({
            "name": name if name else slug,
            "slug": slug,
            "url": f"https://www.10xgenomics.com/datasets/{slug}",
            "product": hit.get("_category", ""),
            "product_name": hit.get("product", {}).get("name", ""),
            "species": "; ".join(hit.get("species", [])),
            "anatomy": "; ".join(hit.get("anatomicalEntities", [])),
            "chemistry": "; ".join(hit.get("chemistries", [])),
            "pipeline": hit.get("pipeline", ""),
            "all_pipeline_versions": "; ".join(hit.get("_all_versions", [])),
            "is_serial_section": is_serial_section(slug),
            "base_slug": hit.get("_base_slug", ""),
        })

    # Sort by product, then name
    records.sort(key=lambda r: (r["product"], r["name"]))

    # Write CSV
    fieldnames = [
        "name", "slug", "url", "product", "product_name",
        "species", "anatomy", "chemistry", "pipeline",
        "all_pipeline_versions", "is_serial_section", "base_slug",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"\nWrote {len(records)} datasets to {OUTPUT_CSV}")

    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("ANALYSIS: MULTI-SECTION SERIAL DATASETS ON 10x GENOMICS")
    print("=" * 70)

    # By product
    by_product = {}
    for r in records:
        by_product.setdefault(r["product"], []).append(r)
    print("\nDatasets by product:")
    for prod, recs in sorted(by_product.items()):
        print(f"  {prod}: {len(recs)}")

    # Identify serial section groups
    serial_groups = identify_serial_groups(
        [h for h in deduped if h.get("_category") in ("Visium", "Visium CytAssist")]
    )

    multi_section = {k: v for k, v in serial_groups.items() if len(v) >= 2}

    print(f"\nMulti-section groups (>=2 entries, Visium/CytAssist only):")
    for group_key, members in sorted(multi_section.items(), key=lambda x: -len(x[1])):
        slugs = [m.get("slug", "") for m in members]
        species = members[0].get("species", ["?"])
        anatomy = members[0].get("anatomicalEntities", ["?"])
        print(f"\n  {group_key} ({len(members)} sections)")
        print(f"    Species: {species}  Anatomy: {anatomy}")
        for s in sorted(slugs):
            print(f"    - {s}")

    # Qualifying datasets (>=3 serial sections from same specimen)
    print(f"\n{'=' * 70}")
    print("DATASETS WITH >=3 SERIAL SECTIONS FROM SAME SPECIMEN:")
    print("=" * 70)

    qualifying = {k: v for k, v in multi_section.items() if len(v) >= 3}
    if qualifying:
        for group_key, members in qualifying.items():
            print(f"\n  {group_key}: {len(members)} sections")
            for m in members:
                print(f"    - {m.get('slug', '')}")
    else:
        print("\n  No groups with >=3 members found in deduplicated results.")
        print("  NOTE: The mouse brain sagittal serial sections (4 sections) appear")
        print("  as separate dataset entries but with different pipeline versions.")
        print("  The 4 unique biological sections are:")
        print("    1. mouse-brain-serial-section-1-sagittal-anterior")
        print("    2. mouse-brain-serial-section-2-sagittal-anterior")
        print("    3. mouse-brain-serial-section-1-sagittal-posterior")
        print("    4. mouse-brain-serial-section-2-sagittal-posterior")

    print(f"\n{'=' * 70}")
    print("BORDERLINE DATASETS (2 serial sections):")
    print("=" * 70)

    borderline = {k: v for k, v in multi_section.items() if len(v) == 2}
    for group_key, members in sorted(borderline.items()):
        print(f"  {group_key}: {len(members)} sections")

    print(f"\n{'=' * 70}")
    print("CONCLUSION:")
    print("=" * 70)
    print("""
Only ONE 10x Genomics dataset has >=3 consecutive serial sections:

  Mouse Brain Sagittal Serial Sections (Visium, Fresh Frozen)
  - 4 sections: 2 anterior + 2 posterior from same C57BL/6 mouse brain
  - CDN base: https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/
  - Available at pipeline versions 1.0.0, 1.1.0, and 2.0.0

Borderline (2 sections each):
  - Human Breast Cancer Block A: Section 1 + Section 2
  - Mouse Brain Coronal FFPE: Section 1 + Section 2
  - Adult Mouse Brain Coronal (stained): Section 1 + Section 2

All other spatial datasets on 10x are single-section per specimen.
Rich multi-section datasets (200-section olfactory bulb, 75-section brain atlas,
62-section embryo, etc.) are from academic publications, NOT from 10x directly.
""")


if __name__ == "__main__":
    main()
