# 🏠 Smart Host Advisor - Intelligent Recommendation System for Airbnb Hosts

## 📋 Overview

**Smart Host Advisor** is an advanced recommendation system designed for Airbnb hosts, providing personalized property improvement recommendations based on deep data analysis. The system analyzes millions of reviews, property features, and product prices to offer hosts accurate recommendations for improving their properties.

---

## 📚 Project Structure

```
smart-host-advisor/
├── Notebooks/                          # Analysis notebooks
│   ├── Airbnb_Reviews_Analysis.ipynb   # Guest reviews analysis
│   ├── Linear_Regression + EDA graps.ipynb  # Linear regression and EDA analysis
│   ├── KNN + Local Analysis + Kalman + Amenities Cost.ipynb  # KNN, Kalman Filter, costs
│   └── Smart host advisor - user interface.ipynb  # Interactive user interface
│
└── scraping/                           # Scraping scripts
    ├── ikea_scraper.py                # IKEA products scraping
    ├── aggregate_prices.py             # Price aggregation by categories
    ├── match_amenities_embeddings.py   # Matching amenities to prices using embeddings
    ├── config.py                       # Proxy and headers configuration
    ├── proxy.py                        # Proxy management
    └── data/                           # Data files
        ├── ikea_products.csv
        ├── ikea_aggregate_amenities.csv
        └── amenity_inventory_with_ikea_price_embeddings.csv
```

---

## 🔬 Notebooks - Detailed Description

### 1. 📝 Airbnb_Reviews_Analysis.ipynb
**Goal:** Identify and classify recurring issues from guest reviews

**Workflow:**
1. **Data Loading:** Load Airbnb data from Italy (98,935 records, 92,941 unique properties)
2. **Review Splitting:** Split review text into individual sentences (3.9M sentences)
3. **Filtering:** 
   - English-only sentences
   - Minimum 5 words
   - Normalization and deduplication
4. **Semantic Issue Detection:** Use MiniLM (sentence-transformers) to identify 21 issue categories:
   - WiFi, TV, Workspace, Climate Control, Furniture, Cleanliness
   - Bathroom, Noise, Space, Kitchen, Rules, Parking, and more...
5. **Sentiment Analysis:** Use SST-2 to identify negative sentences (threshold: 0.40)
6. **Aggregation:** Create aggregation tables at property and host levels
7. **LLM Recommendations:** Use Google Gemini API to generate summaries and practical recommendations

**Outputs:**
- Unique issues tables and their frequency
- Property and host level aggregations
- Evidence tables with example sentences
- Practical LLM recommendations for selected properties

---

### 2. 📊 Linear_Regression + EDA graps.ipynb
**Goal:** Analyze the impact of amenities on price and rating using linear regression

**Workflow:**
1. **Data Loading:** 2,098,880 Airbnb records
2. **Cleaning and Filtering:**
   - Filter properties with valid price (>0)
   - Filter properties with rating
   - Filter properties with minimum 10 reviews (860,729 final properties)
3. **Feature Engineering:**
   - Create binary flags for each amenity (594 common amenities)
   - Create interaction features between amenities
   - Control features: number of guests, beds, superhost, host rating
4. **Models:**
   - **Price Model:** Ridge Regression on `log(price_per_night)`
   - **Rating Model:** Ridge Regression on `ratings` (including log_price as control)
5. **Bootstrap Confidence:** 200 resamples for calculating confidence intervals
6. **Property-Level Recommendations:** Calculate customized recommendations for each property

**Outputs:**
- Coefficient tables for each amenity
- EDA graphs: rating distribution, prices, common amenities
- Property-level recommendations (price_upgrades_all, rating_upgrades_all)
- Tradeoff graphs between price and rating impact

---

### 3. 🔍 KNN + Local Analysis + Kalman + Amenities Cost.ipynb
**Goal:** Local analysis based on KNN and integration with Bayesian model (Kalman Filter)

**Workflow:**

#### Stage 1: KNN - Nearest Neighbors Analysis
1. **Neighbor Calculation:** KNN (K=20) based on:
   - Structural features: bedrooms, beds, bathrooms, guests
   - Location: lat/long (grid-based, GRID_SIZE=0.05)
   - Price (for rating model)
2. **Two Models:**
   - **Price mode:** Distance based on features + location
   - **Rating mode:** Distance based on features + location + price
3. **Amenity Analysis:**
   - Identify common amenities in the neighborhood (≥90% = must-have)
   - Calculate impact on price and rating compared to neighbors

#### Stage 2: Variance Calculation
- Calculate variance of amenity impact on price/rating
- Use variance for confidence calculation

#### Stage 3: Kalman Filter / Bayesian Fusion
1. **Model Integration:**
   - **Prior:** Results from Linear Regression (Format 1)
   - **Measurement:** Results from KNN (Format 2)
2. **Bayesian Update:**
   ```
   K = var_prior / (var_prior + var_knn)
   mu_post = mu_prior + K * (mu_knn - mu_prior)
   var_post = (1 - K) * var_prior
   ```
3. **LCB Score:** Lower Confidence Bound for ranking recommendations
   ```
   score_lcb_log = log(mu_post) - k * sqrt(var_post)
   ```

#### Stage 4: Adding Costs
- Integrate IKEA prices (from `amenity_inventory_with_ikea_price_embeddings.csv`)
- Normalize amenity names for matching

**Outputs:**
- `bayes_price_model`: Bayesian model for price
- `bayes_rating_model`: Bayesian model for rating
- Includes: mu_prior, var_prior, mu_knn, var_knn, mu_post, var_post, score_lcb_log, estimated_cost

---

### 4. Smart host advisor - user interface.ipynb
**Goal:** Interactive user interface for presenting recommendations to hosts

**Features:**
1. **User Input:**
   - Property ID (property number)
   - Max Amenity Cost ($) - optional budget
   - α (price ↔ rating) - weight between price and rating (0.0-1.0)

2. **Bayesian Recommendations:**
   - Top 3 price recommendations (price uplift)
   - Top 3 rating recommendations (rating uplift)
   - Top 3 combined recommendations (combined score)
   - Confidence badges: High/Medium/Low
   - Estimated costs

3. **Market Standard Amenities:**
   - Amenities common in ≥90% of neighboring properties
   - Recommendations for missing amenities

4. **Review Analysis:**
   - LLM summary of issues from reviews
   - Priority badges (P0/P1)
   - List of issues with:
     - Number of mentions
     - Negativity percentage
     - Example sentences

**How it works:**
- All data and models are **precomputed**
- The interface only loads and displays results (very fast)
- Results displayed in styled HTML

---

## 🕷️ Scraping Scripts - IKEA

### 1. ikea_scraper.py
**Goal:** Collect products and prices from IKEA website

**Features:**
- **Crawling:** Automatic traversal of IKEA categories
- **Product Extraction:** Extract name, description, price from each product
- **Resume Support:** Ability to continue scraping from where it stopped
- **Proxy Support:** Use proxy (BrightData) with fallback to direct
- **Persistence:** Save to `ikea_products.csv`

**Usage:**
```python
from scraping.ikea_scraper import crawl_ikea

df = crawl_ikea(
    start_url="https://www.ikea.com/us/en/cat/products-products/",
    max_pages=200,
    resume=True
)
```

---

### 2. aggregate_prices.py
**Goal:** Aggregate IKEA products into amenities with prices

**Process:**
1. **Extract Amenities:** Extract base name from product description (before comma)
2. **Embeddings:** Use MiniLM-L6-v2 to create embeddings
3. **Clustering:** Agglomerative Clustering (distance_threshold=0.15) to group similar products
4. **Aggregation:** Calculate average, minimum, maximum price for each amenity
5. **Filtering:** Only amenities with ≥3 products

**Output:** `ikea_aggregate_amenities.csv` with columns:
- amenity (amenity name)
- estimated_cost (average price)
- min_price, max_price
- n_products

---

### 3. match_amenities_embeddings.py
**Goal:** Match amenities from Airbnb to IKEA prices

**Process:**
1. **Text Normalization:** Normalize amenity names (lowercase, remove punctuation)
2. **Embeddings:** Create embeddings for both sets:
   - Amenities from Airbnb (`amenity_inventory.csv`)
   - Amenities from IKEA (`ikea_aggregate_amenities.csv`)
3. **Similarity Matching:** Calculate cosine similarity and find best match
4. **Confidence Levels:**
   - HIGH: similarity ≥ 0.80
   - MEDIUM: similarity ≥ 0.65
   - NO_MATCH: similarity < 0.65
5. **Manual Rules:** Fallback rules for unmatched cases:
   - Safety: $25
   - Connectivity (WiFi): $0
   - Utilities: $0
   - Bathroom basics: $30
   - Kitchen small: $80
   - Large appliances: $800
   - And more...

**Output:** `amenity_inventory_with_ikea_price_embeddings.csv` with:
- All amenities from Airbnb
- matched_amenity_norm (match from IKEA)
- similarity_score
- match_confidence
- estimated_cost, min_price, max_price

---

## 🔗 How Everything Connects Together

### Complete Pipeline:

```
1. Airbnb Data (Parquet)
   ↓
2. Linear Regression Notebook
   → Prior Model (price_upgrades_all, rating_upgrades_all)
   ↓
3. KNN + Kalman Notebook
   → KNN Model (local analysis)
   → Bayesian Fusion (Prior + KNN)
   → Adding IKEA costs
   ↓
4. Reviews Analysis Notebook
   → Issue detection from reviews
   → LLM recommendations
   ↓
5. User Interface Notebook
   → Load all precomputed models
   → Interactive display for user
```

### Data Flow:

```
IKEA Scraping:
ikea_scraper.py → ikea_products.csv
   ↓
aggregate_prices.py → ikea_aggregate_amenities.csv
   ↓
match_amenities_embeddings.py → amenity_inventory_with_ikea_price_embeddings.csv
   ↓
KNN Notebook (uses costs)

Airbnb Data:
Linear Regression → property_upgrade_recommendations (Prior)
   ↓
KNN Notebook → bayes_price_model, bayes_rating_model (Posterior)
   ↓
Reviews Analysis → airbnb_issues_*, airbnb_api_recs_* (Review insights)
   ↓
User Interface (displays everything together)
```

---

## 🚀 Installation and Running

### System Requirements:
- Python 3.8+
- Apache Spark (with PySpark)
- Azure Databricks (recommended) or local Spark
- Access to Airbnb data (Parquet files)

### Installing Dependencies:

**For Scraping:**
```bash
cd scraping
pip install -r requirements.txt
```

**For Notebooks:**
- PySpark
- sentence-transformers
- transformers
- scikit-learn
- pandas, numpy
- seaborn, matplotlib
- ipywidgets (for user interface)

### Environment Variables Setup:

Create a `.env` file in the `scraping/` directory:
```
BRIGHTDATA_USERNAME=your_username
BRIGHTDATA_PASSWORD=your_password
BRIGHTDATA_HOST=your_host
BRIGHTDATA_PORT=your_port
```

### Running Scraping Scripts:

```bash
# 1. Scrape IKEA products
python scraping/run_ikea_scraper.py

# 2. Aggregate prices
python scraping/aggregate_prices.py

# 3. Match amenities
python scraping/match_amenities_embeddings.py
```

### Running Notebooks:

1. **Upload Data to Databricks:**
   - Upload Airbnb Parquet files
   - Configure SAS token for Azure Storage access

2. **Run Notebooks in Order:**
   - `Linear_Regression + EDA graps.ipynb` (creates Prior)
   - `KNN + Local Analysis + Kalman + Amenities Cost.ipynb` (creates Posterior)
   - `Airbnb_Reviews_Analysis.ipynb` (review analysis - optional)
   - `Smart host advisor - user interface.ipynb` (user interface)

---

## 📊 Usage Examples

### User Interface:

1. Open `Smart host advisor - user interface.ipynb`
2. Enter Property ID (e.g., `1042005770541410920`)
3. Set budget (optional)
4. Choose α (0.0 = rating only, 1.0 = price only)
5. Click "Run Recommendations"
6. View results:
   - Bayesian recommendations with confidence
   - Market standard amenities
   - Review analysis with LLM recommendations


---

## 🔧 Settings and Parameters

### Linear Regression:
- `MIN_REVIEWS = 10` - Minimum reviews per property
- `RIDGE_ALPHAS = [0.1, 1.0, 10.0, 50.0]` - Regularization strengths
- `N_BOOTSTRAP = 200` - Number of resamples for confidence
- `WEIGHT_PRICE = 0.7, WEIGHT_RATING = 0.3` - Weights for combined recommendations

### KNN:
- `N_NEIGHBORS = 20` - Number of neighbors
- `GRID_SIZE = 0.05` - Grid size for location
- `MUST_HAVE_THRESHOLD = 0.9` - Percentage for must-have amenities
- `MIN_SUPPORT = 3` - Minimum support for recommendation

### Reviews Analysis:
- `MIN_WORDS = 5` - Minimum words per sentence
- `ISSUE_SIM_THRESHOLD = 0.4` - Threshold for issue detection
- `NEG_PROB_THRESHOLD = 0.40` - Threshold for negative sentiment

### IKEA Scraping:
- `CLUSTER_DISTANCE_THRESHOLD = 0.15` - Clustering distance
- `MIN_PRODUCTS_PER_AMENITY = 3` - Minimum products per amenity
- `SIMILARITY_THRESHOLD_HIGH = 0.80` - High match
- `SIMILARITY_THRESHOLD_MIN = 0.65` - Minimum match

---

## 📝 Important Notes

1. **Data:** The project requires access to Airbnb data. Files are not included in the repository.

2. **API Keys:** 
   - Google Gemini API key required for LLM recommendations (optional)
   - BrightData credentials required for IKEA scraping

3. **Performance:**
   - Notebooks are designed to run on Spark cluster (Databricks recommended)
   - Full processing of all data can take hours
   - User interface is fast because it loads precomputed data

---



**Built with ❤️ for Airbnb Hosts By Amit&Maayan&Noam&Noa**
