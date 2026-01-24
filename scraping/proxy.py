from scraping.config import (
    BRIGHTDATA_USERNAME,
    BRIGHTDATA_PASSWORD,
    BRIGHTDATA_HOST,
    BRIGHTDATA_PORT
)

def get_proxies():
    """
    Datacenter proxy configuration (allowed by course rules).
    IMPORTANT:
    - BRIGHTDATA_USERNAME must already include the datacenter zone name
      (e.g., brd-customer-XXXX-zone-dc)
    - Do NOT add country / session / residential flags here.
    """
    proxy = (
        f"http://{BRIGHTDATA_USERNAME}:"
        f"{BRIGHTDATA_PASSWORD}"
        f"@{BRIGHTDATA_HOST}:{BRIGHTDATA_PORT}"
    )
    return {
        "http": proxy,
        "https": proxy,
    }