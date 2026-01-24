import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# ==============================
# CONFIG
# ==============================

INPUT_PRODUCTS = "data/ikea_products.csv"
OUTPUT_PATH = "data/ikea_amenities.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Tight threshold → prevents mega-amenities
CLUSTER_DISTANCE_THRESHOLD = 0.15

MIN_PRODUCTS_PER_AMENITY = 3


# ==============================
# HELPERS
# ==============================

def extract_base_amenity(desc: str) -> str:
    """
    Extract the amenity name from product_description.

    Example:
    "Bath towel, dark gray, 28x55 \"" → "bath towel"
    """
    if not isinstance(desc, str):
        return ""

    # Take text before first comma
    base = desc.split(",")[0]

    # Normalize
    base = base.lower()
    base = re.sub(r"[^a-z\s]", "", base)
    base = re.sub(r"\s+", " ", base).strip()

    return base


# ==============================
# MAIN
# ==============================

def main():
    if not os.path.exists(INPUT_PRODUCTS):
        raise FileNotFoundError(INPUT_PRODUCTS)

    print("[LOAD] Reading products...")
    df = pd.read_csv(INPUT_PRODUCTS)

    df = df.dropna(subset=["product_description", "price"])
    df["price"] = df["price"].astype(float)

    print(f"[INFO] Products: {len(df)}")

    # ------------------------------
    # EXTRACT BASE AMENITY
    # ------------------------------
    print("[PARSE] Extracting base amenities...")
    df["base_amenity"] = df["product_description"].apply(extract_base_amenity)

    df = df[df["base_amenity"] != ""]

    print(f"[INFO] Unique base amenities (raw): {df['base_amenity'].nunique()}")

    # ------------------------------
    # EMBEDDINGS (amenity text only)
    # ------------------------------
    print("[EMBED] Encoding base amenities...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    amenity_texts = df["base_amenity"].tolist()
    embeddings = model.encode(
        amenity_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # ------------------------------
    # CLUSTER SIMILAR AMENITIES
    # ------------------------------
    print("[CLUSTER] Clustering similar amenities...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD
    )

    df["amenity_cluster"] = clustering.fit_predict(embeddings)

    print(f"[CLUSTER] Amenity clusters: {df['amenity_cluster'].nunique()}")

    # ------------------------------
    # AGGREGATION
    # ------------------------------
    print("[AGGREGATE] Aggregating amenities...")
    rows = []

    for _, group in df.groupby("amenity_cluster"):
        if len(group) < MIN_PRODUCTS_PER_AMENITY:
            continue

        # Representative amenity name = most common base name
        amenity_name = (
            group["base_amenity"]
            .value_counts()
            .idxmax()
        )

        rows.append({
            "amenity": amenity_name,
            "source": "IKEA",
            "estimated_cost": round(group["price"].median(), 2),
            "min_price": round(group["price"].min(), 2),
            "max_price": round(group["price"].max(), 2),
            "n_products": len(group),
        })

    out = pd.DataFrame(rows).sort_values("n_products", ascending=False)

    print(f"[RESULT] Final amenities: {len(out)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"[SAVE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()