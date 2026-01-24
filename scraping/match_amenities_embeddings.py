import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# CONFIG
# =============================
IKEA_AMENITIES_PATH = "data/ikea_aggregate_amenities.csv"
AMENITY_INVENTORY_PATH = "data/amenity_inventory.csv"
OUTPUT_PATH = "data/amenity_inventory_with_ikea_price_embeddings.csv"

SIMILARITY_THRESHOLD_HIGH = 0.80
SIMILARITY_THRESHOLD_MIN = 0.65

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =============================
# 1. LOAD DATA
# =============================
amenities = pd.read_csv(AMENITY_INVENTORY_PATH)
print(amenities.head())

ikea = pd.read_csv(IKEA_AMENITIES_PATH)

# Required columns:
# amenities: amenity_name
# ikea: amenity_name, estimated_cost, min_price, max_price, n_products

# =============================
# 2. TEXT NORMALIZATION
# =============================
def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

amenities["amenity_norm"] = amenities["amenity_name"].apply(normalize)
ikea["amenity_norm"] = ikea["amenity"].apply(normalize)

# =============================
# 3. LOAD EMBEDDING MODEL
# =============================
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# =============================
# 4. COMPUTE EMBEDDINGS
# =============================
print("Computing embeddings...")
amenity_embeddings = model.encode(
    amenities["amenity_norm"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

ikea_embeddings = model.encode(
    ikea["amenity_norm"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# =============================
# 5. SIMILARITY MATRIX
# =============================
similarity_matrix = cosine_similarity(amenity_embeddings, ikea_embeddings)

best_match_idx = similarity_matrix.argmax(axis=1)
best_match_score = similarity_matrix.max(axis=1)

amenities["matched_amenity_norm"] = ikea.loc[
    best_match_idx, "amenity_norm"
].values

amenities["similarity_score"] = best_match_score

# =============================
# 6. MERGE IKEA PRICES
# =============================
result = amenities.merge(
    ikea[
        [
            "amenity_norm",
            "estimated_cost",
            "min_price",
            "max_price",
            "n_products"
        ]
    ],
    left_on="matched_amenity_norm",
    right_on="amenity_norm",
    how="left",
    suffixes=("", "_ikea")
)

result.drop(columns=["amenity_norm_ikea"], inplace=True)

# =============================
# 7. CONFIDENCE LABEL
# =============================
def confidence_label(score):
    if score >= SIMILARITY_THRESHOLD_HIGH:
        return "HIGH"
    if score >= SIMILARITY_THRESHOLD_MIN:
        return "MEDIUM"
    return "NO_MATCH"

result["match_confidence"] = result["similarity_score"].apply(confidence_label)

# Remove prices for weak matches
mask = result["similarity_score"] < SIMILARITY_THRESHOLD_MIN
price_cols = ["estimated_cost", "min_price", "max_price", "n_products"]
result.loc[mask, price_cols] = np.nan

# =============================
# 7.5 MANUAL RULE-BASED PRICING
# =============================
CATEGORY_RULES = {
    "safety": {
        "keywords": [
            "smoke alarm", "carbon monoxide", "fire extinguisher",
            "first aid", "lock", "security"
        ],
        "price": 25
    },
    "connectivity": {
        "keywords": ["wifi", "internet"],
        "price": 0
    },
    "utilities": {
        "keywords": [
            "hot water", "heating", "air conditioning",
            "ac", "cooling"
        ],
        "price": 0
    },
    "bathroom_basic": {
        "keywords": [
            "towels", "toilet paper", "soap",
            "shampoo", "conditioner", "body soap"
        ],
        "price": 30
    },
    "bedroom_basic": {
        "keywords": [
            "bed linens", "extra pillows", "blankets",
            "hangers", "clothes storage"
        ],
        "price": 40
    },
    "kitchen_small": {
        "keywords": [
            "kettle", "toaster", "coffee maker",
            "microwave", "blender", "rice cooker"
        ],
        "price": 80
    },
    "appliances_large": {
        "keywords": [
            "refrigerator", "fridge", "freezer",
            "oven", "stove", "cooktop", "range",
            "dishwasher", "washing machine", "washer",
            "dryer"
        ],
        "price": 800
    },
    "electronics": {
        "keywords": [
            "tv", "television", "smart tv",
            "cable", "satellite",
            "netflix", "streaming",
            "speaker", "sound system"
        ],
        "price": 350
    },
    "laundry": {
        "keywords": ["washer", "dryer", "laundry"],
        "price": 500
    },
    "workspace": {
        "keywords": ["desk", "chair", "workspace", "office"],
        "price": 120
    },
    "outdoor": {
        "keywords": [
            "balcony", "patio", "garden",
            "outdoor furniture", "bbq"
        ],
        "price": 250
    },
    "family": {
        "keywords": ["crib", "high chair", "baby"],
        "price": 120
    }
}

def apply_manual_category_price(row):
    if row["match_confidence"] != "NO_MATCH":
        return row

    name = row["amenity_norm"]
    for category, rule in CATEGORY_RULES.items():
        for kw in rule["keywords"]:
            if kw in name:
                price = rule["price"]
                row["estimated_cost"] = price
                row["min_price"] = price
                row["max_price"] = price
                row["n_products"] = 1
                row["match_confidence"] = f"MANUAL_{category.upper()}"
                return row
    return row

result = result.apply(apply_manual_category_price, axis=1)

# =============================
# 8. SAVE RESULT
# =============================
result.to_csv(OUTPUT_PATH, index=False)

# =============================
# 9. REPORT
# =============================
print("\n=== EMBEDDING MATCHING REPORT ===")
print(f"Total amenities: {len(result)}")
print(result["match_confidence"].value_counts())

print("\nSample HIGH confidence matches:")
print(
    result[result["match_confidence"] == "HIGH"][
        ["amenity_name", "matched_amenity_norm", "similarity_score"]
    ].head(5)
)

print("\nSample NO_MATCH:")
print(
    result[result["match_confidence"] == "NO_MATCH"][
        ["amenity_name", "matched_amenity_norm", "similarity_score"]
    ].head(5)
)

# =============================
# 10. PRICE COVERAGE REPORT
# =============================
total = len(result)
with_price = result["estimated_cost"].notna().sum()
without_price = result["estimated_cost"].isna().sum()

print("\n=== PRICE COVERAGE REPORT ===")
print(f"Total amenities: {total}")
print(f"Amenities with matched price: {with_price} ({with_price / total:.1%})")
print(f"Amenities without matched price: {without_price} ({without_price / total:.1%})")

# =============================
# 11. MATCH SOURCE BREAKDOWN
# =============================

total = len(result)

ikea_matched = result[
    result["match_confidence"].isin(["HIGH", "MEDIUM"])
].shape[0]

manual_matched = result[
    result["match_confidence"].str.startswith("MANUAL_", na=False)
].shape[0]

no_match = result[
    result["match_confidence"] == "NO_MATCH"
].shape[0]

print("\n=== MATCH SOURCE BREAKDOWN ===")
print(f"Total amenities: {total}")
print(f"IKEA-based matches: {ikea_matched} ({ikea_matched / total:.1%})")
print(f"Manual rule-based matches: {manual_matched} ({manual_matched / total:.1%})")
print(f"Unmatched amenities: {no_match} ({no_match / total:.1%})")