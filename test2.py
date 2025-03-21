# scraper.py
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
from Levenshtein import ratio as levenshtein_ratio

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

class ScraperConfig(BaseModel):
    base_url: str = "https://www.heinemann-shop.com"
    default_airport: str = "fra"
    search_path: str = "/en/{airport}/search/"
    query_param: str = "text"
    request_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    rate_limit_delay: float = 1.0
    user_agent: str = "Mozilla/5.0 (compatible; ScraperBot/1.0; +https://example.com/scraper)"
    headers: Dict = {
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.heinemann-shop.com/",
    }

class Selectors(BaseModel):
    results_count: str = "h1.m-category-overview__title"
    product_list: str = "ul.c-article-list"
    product_item: str = "li.c-article-list__item"
    json_script: str = "script[type='application/json']"
    brand: str = "span[itemprop='brand']"
    price: str = "div.c-article-tile__price"
    image: str = "img.c-article-list-item__image"
    product_link: str = "a.js-ga-article-link"
    product_description: str = "div.c-accordion__content.js-accordion-content p"
    product_details_table: str = "table.c-product-details-table"
    detail_row: str = "tr.c-product-details-table__row"
    detail_key: str = "td.c-product-details-table__key"
    detail_value: str = "td.c-product-details-table__value"

class ProductData(BaseModel):
    id: str
    name: str
    price: str
    currency: str
    brand: str
    category: str
    variant: str
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    airport: str
    raw_data: Optional[Dict] = None
    description: Optional[str] = None
    item_number: Optional[str] = None
    alcohol_by_volume: Optional[float] = None
    manufacturer_info: Optional[str] = None
    warnings: Optional[str] = None
    specifications: Optional[Dict[str, str]] = None

class HeinemannScraper:
    def __init__(self, config: ScraperConfig = None, selectors: Selectors = None):
        self.config = config or ScraperConfig()
        self.selectors = selectors or Selectors()
        self.session = self._configure_session()
        self.logger = self._configure_logger()
        self.current_product_count = 0

    def _configure_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.config.user_agent})
        session.headers.update(self.config.headers)
        return session

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _construct_url(self, airport: str, search_text: str, page: int = None) -> str:
        base_path = self.config.search_path.format(airport=airport)
        url = urljoin(self.config.base_url, base_path)
        params = {self.config.query_param: search_text}
        if page is not None and page > 0:
            params["page"] = page
        return requests.Request("GET", url, params=params).prepare().url

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            time.sleep(self.config.rate_limit_delay)
            return BeautifulSoup(response.content, "html.parser")
        except RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return None

    def _parse_results_count(self, soup: BeautifulSoup) -> int:
        element = soup.select_one(self.selectors.results_count)
        return int(element.text.split()[0]) if element else 0

    def _passes_filters(self, product_data: ProductData, search_text: str, category: str = None) -> bool:
        def normalize(text):
            return text.lower().replace('-', ' ').translate(str.maketrans('', '', ',.'))

        required_keywords = [normalize(kw) for kw in search_text.split() if kw]
        product_name = normalize(product_data.name)
        
        if category and normalize(product_data.category) != normalize(category):
            return False

        for kw in required_keywords:
            if not any(
                levenshtein_ratio(kw, name_part) >= 0.85 or 
                kw in name_part
                for name_part in product_name.split()
            ):
                self.logger.info(f"Rejected: Missing keyword '{kw}' in '{product_name}'")
                return False

        product_brand = normalize(product_data.brand or "")
        if not any(levenshtein_ratio(kw, product_brand) >= 0.8 for kw in required_keywords):
            self.logger.info(f"Rejected: Brand mismatch '{product_brand}'")
            return False

        name_similarity = levenshtein_ratio(normalize(search_text), product_name)
        if name_similarity < 0.7:
            self.logger.info(f"Rejected: Low overall similarity ({name_similarity:.0%})")
            return False

        return True

    def _extract_product_data(self, item: BeautifulSoup, airport: str) -> Optional[ProductData]:
        self.current_product_count += 1
        product_log_prefix = f"Product #{self.current_product_count}"

        try:
            script = item.select_one(self.selectors.json_script)
            if not script:
                return None

            try:
                json_data = json.loads(script.text.strip())
            except json.JSONDecodeError:
                return None

            product_url = item.select_one(self.selectors.product_link)
            if not product_url or 'href' not in product_url.attrs:
                return None

            image = item.select_one(self.selectors.image)
            image_url = image.get("src") if image else None

            product_data = {
                "id": json_data.get("id"),
                "name": json_data.get("name"),
                "price": json_data.get("price"),
                "currency": "EUR",
                "brand": json_data.get("brand"),
                "category": json_data.get("category"),
                "variant": json_data.get("dimension3", ""),
                "image_url": image_url,
                "product_url": urljoin(self.config.base_url, product_url["href"]),
                "airport": airport,
                "raw_data": json_data,
            }

            return ProductData(**product_data)
        except (AttributeError, KeyError, ValidationError):
            return None

    def _scrape_product_details(self, product_url: str) -> Dict:
        details = {}
        try:
            soup = self._get_page(product_url)
            if not soup:
                return details

            description_div = soup.select_one("div.c-accordion__content.js-accordion-content")
            if description_div:
                description_parts = [p.text.strip() for p in description_div.select("p")]
                details['description'] = ' '.join(description_parts)

            specs = {}
            table = soup.select_one("table.c-product-details-table")
            if table:
                rows = table.select("tr.c-product-details-table__row")
                for row in rows:
                    key_element = row.select_one("td.c-product-details-table__key")
                    value_element = row.select_one("td.c-product-details-table__value")
                    if key_element and value_element:
                        specs[key_element.text.strip()] = value_element.text.strip()

                details.update({
                    'item_number': specs.get('Item No.'),
                    'alcohol_by_volume': float(specs.get('Alcohol by volume [% AbV]', 0)) if specs.get('Alcohol by volume [% AbV]') else None,
                    'manufacturer_info': specs.get('Manufacturer information'),
                    'warnings': specs.get('Warnings'),
                    'specifications': specs
                })

            return details
        except Exception:
            return {}

    def scrape(
        self,
        search_text: str,
        airport: str = None,
        max_products: Optional[int] = None,
        category: Optional[str] = None
    ) -> Tuple[int, List[ProductData]]:
        airport = airport or self.config.default_airport
        products = []
        page = 0
        total_pages_processed = 0
        consecutive_empty_pages = 0
        should_continue = True

        while should_continue:
            url = self._construct_url(airport, search_text, page)
            self.logger.info(f"Processing page {page + 1} ({url})")

            soup = self._get_page(url)
            if not soup:
                break

            product_list = soup.select(self.selectors.product_list)
            page_products = []
            skipped_count = 0
            consecutive_skips = 0

            for list_idx, pl in enumerate(product_list, 1):
                items = pl.select(self.selectors.product_item)
                for item in items:
                    if max_products and len(products) >= max_products:
                        break

                    product_data = self._extract_product_data(item, airport)
                    if not product_data:
                        skipped_count += 1
                        consecutive_skips += 1
                        continue

                    if self._passes_filters(product_data, search_text, category):
                        consecutive_skips = 0
                        details = self._scrape_product_details(product_data.product_url)
                        page_products.append(product_data.model_copy(update=details))
                    else:
                        skipped_count += 1
                        consecutive_skips += 1

                    if consecutive_skips >= 10:
                        self.logger.info("Stopping due to 10 consecutive skips")
                        should_continue = False
                        break

                if max_products and len(products) >= max_products:
                    break

            products.extend(page_products)
            total_pages_processed += 1

            # Update pagination logic
            if total_pages_processed < 1:
                page += 1
                should_continue = True
            else:
                if len(page_products) > 0:
                    page += 1
                    consecutive_empty_pages = 0
                else:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= 1:
                        should_continue = False

            if skipped_count > len(page_products) * 3:
                self.logger.info("Stopping due to high skip ratio")
                should_continue = False

        self.logger.info(f"Scraped {len(products)} products from {total_pages_processed} pages")
        return len(products), products
    
    def print_scraped_data(self, products: List[ProductData]):
        """Prints all scraped product data in a readable format."""
        if not products:
            print("No products found.")
            return

        print("\n" + "=" * 50)
        print("Scraped Product Data")
        print("=" * 50)

        for idx, product in enumerate(products, 1):
            print(f"\nProduct #{idx}:")
            print(f"  ID: {product.id}")
            print(f"  Name: {product.name}")
            print(f"  Brand: {product.brand}")
            print(f"  Category: {product.category}")
            print(f"  Price: {product.price} {product.currency}")
            print(f"  URL: {product.product_url}")
            print(f"  Image URL: {product.image_url}")
            print(f"  Description: {product.description}")
            print(f"  Item Number: {product.item_number}")
            print(f"  Alcohol by Volume: {product.alcohol_by_volume}")
            print(f"  Manufacturer Info: {product.manufacturer_info}")
            print(f"  Warnings: {product.warnings}")
            if product.specifications:
                print("  Specifications:")
                for key, value in product.specifications.items():
                    print(f"    {key}: {value}")
            print("-" * 50)

    
scraper = HeinemannScraper()
results_count, products = scraper.scrape(
    search_text="Jonny blue walker",
    category="whiskey",  # Optional category filter
    max_products=100
)

scraper.print_scraped_data(products)
