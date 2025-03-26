import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs  # Added parse_qs import

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
    max_pages: int = 10
    enable_deep_scrape: bool = True
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
    pagination: str = "ul.c-pagination"
    pagination_item: str = "li.c-pagination__page a"
    next_page: str = "li.c-pagination__next a"


class ProductData(BaseModel):
    id: str
    name: str
    price: str
    currency: str
    brand: str
    category: str
    variant: str
    image_url: Optional[str]
    product_url: Optional[str]
    airport: str
    raw_data: Optional[Dict]
    item_number: Optional[str]
    alcohol_by_volume: Optional[float]  # Correct type is float
    manufacturer_info: Optional[str]
    warnings: Optional[str]
    description: Optional[str]
    page_number: Optional[int]
    specifications: Optional[Dict[str, str]]


class HeinemannScraper:
    def __init__(self, config: ScraperConfig = None, selectors: Selectors = None):
        self.config = config or ScraperConfig()
        self.selectors = selectors or Selectors()
        self.session = self._configure_session()
        self.logger = self._configure_logger()
        self.current_page = 1

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
        params = {self.config.query_param: search_text.replace(" ", "+")}
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
        try:
            script = item.select_one(self.selectors.json_script)
            json_data = json.loads(script.text.strip()) if script else {}

            product_url = item.select_one(self.selectors.product_link)
            image = item.select_one(self.selectors.image)
            price_element = item.select_one(self.selectors.price)
            brand_element = item.select_one(self.selectors.brand)

            # Ensure all required fields are populated
            product_data = {
                "id": json_data.get("id", ""),  
                "name": json_data.get("name", ""),  
                "price": price_element.text.strip() if price_element else "0.00",  
                "currency": "EUR",  # Hardcoded as per your requirements
                "brand": brand_element.text.strip() if brand_element else "Unknown", 
                "category": json_data.get("category", ""), 
                "variant": json_data.get("dimension3", ""),  
                "image_url": image.get("src") if image else None,
                "product_url": urljoin(self.config.base_url, product_url["href"]) if product_url else None,
                "airport": airport,  
                "raw_data": json_data,
            }

            return ProductData(**product_data)
        except (AttributeError, KeyError, ValidationError) as e:
            self.logger.warning(f"Failed to parse product: {str(e)}")
            return None

    def _handle_pagination(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all pagination links"""
        pagination_links = []
        
        pagination = soup.select_one(self.selectors.pagination)
        if pagination:
            page_links = pagination.select(self.selectors.pagination_item)
            for link in page_links:
                href = link.get('href')
                if href:
                    pagination_links.append(urljoin(base_url, href))
            
            next_page = pagination.select_one(self.selectors.next_page)
            if next_page and next_page.get('href'):
                pagination_links.append(urljoin(base_url, next_page['href']))
        
        unique_pages = list(set(pagination_links))
        return sorted(unique_pages, key=lambda x: int(parse_qs(urlparse(x).query).get('page', ['0'])[0]))

    def _scrape_product_page(self, product_url: str) -> Dict:
        """Scrape detailed product information from individual product page"""
        details = {}
        try:
            soup = self._get_page(product_url)
            if not soup:
                return details

            description = ' '.join([p.text.strip() for p in 
                                  soup.select(self.selectors.product_description)])
            
            specs = {}
            table = soup.select_one(self.selectors.product_details_table)
            if table:
                for row in table.select('tr.c-product-details-table__row'):
                    key_elem = row.select_one('.c-product-details-table__key')
                    value_elem = row.select_one('.c-product-details-table__value')
                    if key_elem and value_elem:
                        key = key_elem.text.strip()
                        value = value_elem.text.strip()
                        specs[key] = value

            # Process alcohol_by_volume
            abv_str = specs.get('Alcohol by volume [% AbV]')
            abv = None
            if abv_str:
                try:
                    abv_clean = abv_str.split('%')[0].strip()
                    abv = float(abv_clean)
                except ValueError:
                    pass

            details.update({
                'description': description,
                'item_number': specs.get('Item No.'),
                'alcohol_by_volume': abv,
                'manufacturer_info': specs.get('Manufacturer information'),
                'warnings': specs.get('Warnings'),
                'specifications': specs
            })
        except Exception as e:
            self.logger.error(f"Failed to scrape product page {product_url}: {str(e)}")
        return details

    def scrape(self, search_text: str, airport: str = None) -> Tuple[int, List[ProductData]]:
        airport = airport or self.config.default_airport
        base_url = self._construct_url(airport, search_text)
        all_products = []
        current_url = base_url
        page_count = 0

        while current_url and page_count < self.config.max_pages:
            self.logger.info(f"Scraping page {page_count + 1}")
            soup = self._get_page(current_url)
            if not soup:
                break

            if page_count == 0:
                pagination_links = self._handle_pagination(soup, base_url)
                self.logger.debug(f"Pagination links: {pagination_links}")

            product_list = soup.select(self.selectors.product_list)
            for product in product_list:
                for item in product.select(self.selectors.product_item):
                    product_data = self._extract_product_data(item, airport)
                    if product_data and self.config.enable_deep_scrape:
                        details = self._scrape_product_page(product_data.product_url)
                        product_data = product_data.copy(update=details)
                    if product_data:
                        product_data.page_number = page_count + 1
                        all_products.append(product_data)

            current_url = self._get_next_page(pagination_links, page_count)
            page_count += 1

        return len(all_products), all_products

    def _get_next_page(self, pagination_links: List[str], current_page: int) -> Optional[str]:
        """Determine next page URL from pagination links"""
        if current_page + 1 < len(pagination_links):
            return pagination_links[current_page + 1]
        return None
if __name__ == "__main__":
    config = ScraperConfig(
        max_pages=3,
        enable_deep_scrape=True,
        rate_limit_delay=2.0
    )
    
    scraper = HeinemannScraper(config=config)
    total, products = scraper.scrape("johnny blue whiskey", "fra")  

    print(f"Total products scraped: {total}")
    for product in products[:2]:
        print(f"\nProduct: {product.name}")
        print(f"Description: {product.description[:100]}..." if product.description else "No description")
        print(f"Specifications: {json.dumps(product.specifications, indent=2)}")