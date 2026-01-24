from scraping.ikea_scraper import crawl_ikea

START_URL = "https://www.ikea.com/us/en/cat/products-products/"


# START_URL = "https://www.ikea.com/us/en/cat/home-electronics-he001/"


def main():
    df = crawl_ikea(START_URL, max_pages=10000, resume=True)

    print(df.head())
    print(f"Total products scraped: {len(df)}")

    df.to_csv("data/ikea_products_final.csv", index=False)

if __name__ == "__main__":
    main()