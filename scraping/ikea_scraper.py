import random
import time
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

from scraping.proxy import get_proxies
from scraping.config import HEADERS


# ---------- counted urls persistence ----------
COUNTED_URLS_PATH = "data/ikea_counted_urls.csv"
PRODUCTS_PATH = "data/ikea_products.csv"

# Helper to load visited URLs from ikea_products.csv
def load_visited_urls_from_products():
    if not os.path.exists(PRODUCTS_PATH):
        return set()
    df = pd.read_csv(PRODUCTS_PATH, usecols=["category_url"])
    return set(df["category_url"].dropna().astype(str))

def load_counted_urls():
    if not os.path.exists(COUNTED_URLS_PATH):
        return set()
    df = pd.read_csv(COUNTED_URLS_PATH)
    return set(df["url"].astype(str).tolist())

def append_counted_url(url):
    df = pd.DataFrame({"url": [url]})
    df.to_csv(
        COUNTED_URLS_PATH,
        mode="a",
        header=not os.path.exists(COUNTED_URLS_PATH),
        index=False,
    )

# ---------- products persistence ----------

def reset_products_file():
    if os.path.exists(PRODUCTS_PATH):
        os.remove(PRODUCTS_PATH)

def append_products(rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(PRODUCTS_PATH), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(
        PRODUCTS_PATH,
        mode="a",
        header=not os.path.exists(PRODUCTS_PATH),
        index=False,
    )


BASE_URL = "https://www.ikea.com"


# ---------- session + fetch ----------

DEFAULT_TIMEOUT = (5, 25)  # (connect, read)
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session

    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    _session = s
    return _session


def fetch_soup(url):
    time.sleep(random.uniform(0.3, 1.0))  # polite jitter
    s = _get_session()

    # The root IKEA category page is very heavy and often hangs on DC proxies.
    # Fetch it directly (no proxy) to avoid blocking the crawl.
    if "products-products" in url:
        r = s.get(
            url,
            headers=HEADERS,
            timeout=DEFAULT_TIMEOUT,
            verify=False,
        )
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")

    # --- attempt with proxy ---
    try:
        r = s.get(
            url,
            headers=HEADERS,
            proxies=get_proxies(),
            timeout=DEFAULT_TIMEOUT,
            verify=False,
        )
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")

    except (requests.exceptions.ProxyError,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout) as e:
        print(f"[WARN] Proxy failed for {url} ({type(e).__name__}), retrying direct...")

    # --- fallback: direct request (no proxy) ---
    r = s.get(
        url,
        headers=HEADERS,
        timeout=DEFAULT_TIMEOUT,
        verify=False,
    )
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


# ---------- page helpers ----------

def is_product_page(soup):
    return len(soup.select("div.plp-mastercard")) > 0


def get_subcategory_links(soup):
    links = set()
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue

        full = urljoin(BASE_URL, href)

        if full.startswith("https://www.ikea.com/us/en/cat/") and full.endswith("/"):
            links.add(full)

    return links


# ---------- product scraping ----------

def scrape_products_from_plp(url, category_name):
    soup = fetch_soup(url)
    products = []

    cards = soup.select("div.plp-mastercard")
    print(f"[PLP] {url} → {len(cards)} products")

    for card in cards:
        name = card.select_one("span.plp-price-module__product-name")
        desc = card.select_one("span.plp-price-module__description")
        price_int = card.select_one("span.plp-price__integer")
        price_dec = card.select_one("span.plp-price__decimal")

        if not name or not price_int:
            continue

        price = price_int.text.strip()
        if price_dec:
            price += price_dec.text.strip()

        try:
            price = float(price)
        except ValueError:
            continue

        products.append({
            "product_name": name.text.strip(),
            "product_description": desc.text.strip() if desc else "",
            "product_category": category_name,
            "price": price,
            "source": "IKEA",
            "category_url": url
        })

    return products


# ---------- main crawler ----------

def crawl_ikea(start_url, max_pages=200, resume=True):
    if not resume:
        reset_products_file()
        visited_plps = set()
        print("[INIT] Starting fresh run, cleared ikea_products.csv")
    else:
        visited_plps = load_visited_urls_from_products()

        if start_url in visited_plps:
            visited_plps.remove(start_url)
            print(f"[INIT] Removed start_url from visited PLPs to allow traversal: {start_url}")

        print(f"[INIT] Loaded {len(visited_plps)} scraped PLP URLs from ikea_products.csv")

    visited_pages = set()

    stack = [start_url]
    # visited_pages.add(start_url)
    pages_crawled = 0
    while stack and pages_crawled < max_pages:
        url = stack.pop()

        if url in visited_pages:
            continue

        visited_pages.add(url)
        pages_crawled += 1
        print(f"[{pages_crawled}] Crawling: {url}")

        print(f"[FETCH] {url}")

        try:
            soup = fetch_soup(url)
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on {url}, skipping")
            continue
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Request failed on {url}: {e}")
            continue

        if is_product_page(soup):
            if url in visited_plps:
                pages_crawled -= 1  # don't count already scraped PLPs
                print(f"[SKIP] PLP already scraped: {url}")
                continue

            try:
                products = scrape_products_from_plp(url, category_name="IKEA")
                if products:
                    append_products(products)
                    visited_plps.add(url)
                    print(f"[SAVE] Saved {len(products)} products from {url}")
            except requests.exceptions.RequestException as e:
                print(f"[WARN] Failed scraping PLP {url}: {e}")
            continue

        for sub in get_subcategory_links(soup):
            if sub not in visited_pages and sub not in stack:
                stack.append(sub)

    if os.path.exists(PRODUCTS_PATH):
        return pd.read_csv(PRODUCTS_PATH)
    return pd.DataFrame()


# ---------- page counting ----------

def count_ikea_pages(start_url, max_pages=1000):
    os.makedirs(os.path.dirname(COUNTED_URLS_PATH), exist_ok=True)

    visited = set()
    stack = [start_url]

    total_pages = 0
    product_pages = 0
    category_pages = 0

    rows = []

    while stack and len(visited) < max_pages:
        url = stack.pop()

        if url in visited:
            continue

        visited.add(url)
        total_pages += 1
        print(f"[{total_pages}] Visiting: {url}")

        try:
            soup = fetch_soup(url)
        except Exception as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            continue

        if is_product_page(soup):
            page_type = "product"
            product_pages += 1
        else:
            page_type = "category"
            category_pages += 1

            subcats = get_subcategory_links(soup)
            for sub in subcats:
                if sub not in visited:
                    stack.append(sub)

        rows.append({
            "url": url,
            "page_type": page_type
        })

        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df.to_csv(COUNTED_URLS_PATH, index=False)

    return {
        "total_pages": total_pages,
        "product_pages": product_pages,
        "category_pages": category_pages,
        "unique_pages": len(visited),
        "csv_path": COUNTED_URLS_PATH
    }