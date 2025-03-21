# scraper.py
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

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

    def _construct_url(self, airport: str, search_text: str) -> str:
        base_path = self.config.search_path.format(airport=airport)
        url = urljoin(self.config.base_url, base_path)
        params = {self.config.query_param: search_text}
        return requests.Request("GET", url, params=params).prepare().url

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(
                url, timeout=self.config.request_timeout
            )
            response.raise_for_status()
            time.sleep(self.config.rate_limit_delay)
            return BeautifulSoup(response.content, "html.parser")
        except RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return None

    def _parse_results_count(self, soup: BeautifulSoup) -> int:
        element = soup.select_one(self.selectors.results_count)
        return int(element.text.split()[0]) if element else 0

    def _extract_product_data(self, item: BeautifulSoup, airport: str) -> Optional[ProductData]:
        self.current_product_count += 1
        product_log_prefix = f"Product #{self.current_product_count}"

        try:
            self.logger.info(f"{product_log_prefix} - Starting extraction")

            # JSON script extraction
            script = item.select_one(self.selectors.json_script)
            if not script:
                self.logger.warning(f"{product_log_prefix} - Missing JSON script tag")
                return None

            try:
                json_data = json.loads(script.text.strip())
                self.logger.debug(f"{product_log_prefix} - JSON data parsed successfully")
            except json.JSONDecodeError:
                self.logger.error(f"{product_log_prefix} - Invalid JSON data in script tag")
                return None

            # URL and image handling
            product_url = item.select_one(self.selectors.product_link)
            if not product_url or 'href' not in product_url.attrs:
                self.logger.warning(f"{product_log_prefix} - Missing product URL")
                return None

            image = item.select_one(self.selectors.image)
            image_url = image.get("src") if image else None
            if not image_url:
                self.logger.debug(f"{product_log_prefix} - No image URL found")

            product_data = {
                "id": json_data.get("id"),
                "name": json_data.get("name"),
                "price": json_data.get("price"),
                "currency": "EUR",
                "brand": json_data.get("brand"),
                "category": json_data.get("category"),
                "variant": json_data.get("dimension3", ""),
                "image_url": image_url,
                "product_url": urljoin(self.config.base_url, product_url["href"]) if product_url else None,
                "airport": airport,
                "raw_data": json_data,
            }

            self.logger.info(f"{product_log_prefix} - Base data extracted successfully")
            return ProductData(**product_data)

        except (AttributeError, KeyError, ValidationError) as e:
            self.logger.error(f"{product_log_prefix} - Extraction failed: {str(e)}")
            return None

    def _scrape_product_details(self, product_url: str) -> Dict:
        """Scrape detailed product information from individual product page"""
        self.logger.info(f"Scraping product details: {product_url}")
        details = {}

        try:
            soup = self._get_page(product_url)
            if not soup:
                self.logger.warning(f"Details failed - Empty page for {product_url}")
                return details

            # Description extraction
            description_div = soup.select_one("div.c-accordion__content.js-accordion-content")
            if description_div:
                try:
                    description_parts = [p.text.strip() for p in description_div.select("p")]
                    details['description'] = ' '.join(description_parts)
                    self.logger.debug(f"Description found ({len(description_parts)} parts)")
                except Exception as e:
                    self.logger.warning(f"Description extraction failed: {str(e)}")
            else:
                self.logger.debug("No description section found")

            # Specifications table
            specs = {}
            table = soup.select_one("table.c-product-details-table")
            if table:
                self.logger.debug("Processing specifications table")
                rows = table.select("tr.c-product-details-table__row")
                self.logger.info(f"Found {len(rows)} specification rows")

                for i, row in enumerate(rows, 1):
                    try:
                        key_element = row.select_one("td.c-product-details-table__key")
                        value_element = row.select_one("td.c-product-details-table__value")
                        if key_element and value_element:
                            key = key_element.text.strip()
                            value = value_element.text.strip()
                            specs[key] = value
                            self.logger.debug(f"Row {i}: {key} = {value}")
                        else:
                            self.logger.debug(f"Row {i}: Incomplete row skipped")
                    except Exception as e:
                        self.logger.warning(f"Row {i} processing failed: {str(e)}")

                details.update({
                    'item_number': specs.get('Item No.'),
                    'alcohol_by_volume': float(specs.get('Alcohol by volume [% AbV]', 0)) if specs.get('Alcohol by volume [% AbV]') else None,
                    'manufacturer_info': specs.get('Manufacturer information'),
                    'warnings': specs.get('Warnings'),
                    'specifications': specs
                })
            else:
                self.logger.debug("No specifications table found")

            return details

        except Exception as e:
            self.logger.error(f"Product details failed for {product_url}: {str(e)}")
            return {}

    def scrape(
        self,
        search_text: str,
        airport: str = None,
        max_products: Optional[int] = None
    ) -> Tuple[int, List[ProductData]]:
        """
        Scrape products sequentially.

        Args:
            search_text: The search query.
            airport: The airport code (defaults to config value).
            max_products: Maximum number of products to scrape. If None, scrape all.

        Returns:
            Tuple of (total results count, list of scraped products).
        """
        self.logger.info("=" * 50)
        self.logger.info(f"Starting new scrape session for: {search_text}")
        self.logger.info(f"Max products to scrape: {max_products if max_products else 'All'}")
        self.logger.info("=" * 50)

        airport = airport or self.config.default_airport
        url = self._construct_url(airport, search_text)
        self.logger.info(f"Constructed target URL: {url}")

        soup = self._get_page(url)
        if not soup:
            self.logger.error("Initial page fetch failed - aborting")
            return 0, []

        results_count = self._parse_results_count(soup)
        self.logger.info(f"Search reported {results_count} results")

        product_list = soup.select(self.selectors.product_list)
        self.logger.info(f"Found {len(product_list)} product list containers")

        products = []
        for list_idx, product_list in enumerate(product_list, 1):
            items = product_list.select(self.selectors.product_item)
            self.logger.info(f"Processing list {list_idx} with {len(items)} items")

            for item_idx, item in enumerate(items, 1):
                self.logger.info("-" * 40)
                self.logger.info(f"Processing item {item_idx}/{len(items)} in list {list_idx}")

                # Stop if max_products is reached
                if max_products and len(products) >= max_products:
                    self.logger.info(f"Stopping early: reached max_products limit ({max_products})")
                    break

                product_data = self._extract_product_data(item, airport)
                if not product_data:
                    self.logger.warning(f"Skipping item {item_idx} due to extraction errors")
                    continue

                try:
                    self.logger.info(f"Fetching details for: {product_data.product_url}")
                    details = self._scrape_product_details(product_data.product_url)
                    products.append(product_data.model_copy(update=details))
                    self.logger.info(f"Product {len(products)} completed successfully")
                except Exception as e:
                    self.logger.error(f"Details processing failed: {str(e)}")
                    products.append(product_data)  # Keep base data if available

        self.logger.info("=" * 50)
        self.logger.info(f"Scrape completed. Total products processed: {len(products)}")
        self.logger.info("=" * 50)

        return results_count, products