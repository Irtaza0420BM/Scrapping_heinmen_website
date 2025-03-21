import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin
import re
from difflib import SequenceMatcher
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError, validator
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
from openai import OpenAI
import csv
import os
from datetime import datetime


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
    headers: Dict[str, str] = Field(default_factory=lambda: {
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.heinemann-shop.com/",
    })
    api_key: Optional[str] = None
    log_level: int = logging.INFO
    output_dir: str = "output"


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
    currency: str = "EUR"
    brand: str
    category: str
    variant: str = ""
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    airport: str
    description: Optional[str] = None
    item_number: Optional[str] = None
    alcohol_by_volume: Optional[float] = None
    manufacturer_info: Optional[str] = None
    warnings: Optional[str] = None
    specifications: Optional[Dict[str, str]] = None
    ai_extracted_data: Optional[Dict[str, Any]] = None
    scrape_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Raw data is removed from the model to avoid redundancy

    @validator('name', 'brand', 'category', 'variant', 'description', 'manufacturer_info', 'warnings', pre=True)
    def clean_text_fields(cls, v):
        """Clean text fields to handle string issues."""
        if v is None:
            return None
        if isinstance(v, str):
            # Remove extra whitespace, normalize line breaks
            cleaned = re.sub(r'\s+', ' ', v).strip()
            # Remove any control characters
            cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
            return cleaned
        return v
    
    @validator('price', pre=True)
    def clean_price(cls, v):
        """Clean price field to ensure consistent format."""
        if v is None:
            return None
        if isinstance(v, str):
            # Extract numbers and decimal point only
            price_match = re.search(r'(\d+[.,]?\d*)', v)
            if price_match:
                return price_match.group(1).replace(',', '.')
        return v
    
    @validator('alcohol_by_volume', pre=True)
    def clean_abv(cls, v):
        """Convert alcohol percentage to float."""
        if v is None:
            return None
        if isinstance(v, str):
            # Extract percentage value
            abv_match = re.search(r'(\d+[.,]?\d*)', v)
            if abv_match:
                try:
                    return float(abv_match.group(1).replace(',', '.'))
                except ValueError:
                    return None
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a 4D-compatible dictionary."""
        data = self.dict(exclude_none=True)
        
        # Format specifications as a single string for 4D compatibility
        if self.specifications:
            spec_str = "; ".join(f"{k}: {v}" for k, v in self.specifications.items())
            data["specifications_text"] = spec_str
        
        if self.ai_extracted_data:
            data["ai_data_json"] = json.dumps(self.ai_extracted_data)
            
        return data


class HeinemannScraper:
    
    def __init__(self, config: Optional[ScraperConfig] = None, selectors: Optional[Selectors] = None):
        self.config = config or ScraperConfig()
        self.selectors = selectors or Selectors()
        self.session = self._configure_session()
        self.logger = self._configure_logger()
        self.current_product_count = 0
        self.openai_client = self._configure_openai()
        
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _configure_session(self) -> requests.Session:
        """Configure HTTP session with retries and headers."""
        session = requests.Session()
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.config.user_agent})
        session.headers.update(self.config.headers)
        return session

    def _configure_logger(self) -> logging.Logger:
        """Configure logger with appropriate handlers and formatters."""
        logger = logging.getLogger("heinemann_scraper")
        
        if logger.handlers:
            logger.handlers.clear()
            
        logger.setLevel(self.config.log_level)
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(os.path.join(self.config.output_dir, "scraper.log"))
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _configure_openai(self) -> Optional[OpenAI]:
        if self.config.api_key:
            try:
                return OpenAI(api_key=self.config.api_key)
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                return None
        else:
            self.logger.warning("No OpenAI API key provided. AI extraction will be skipped.")
            return None

    def _construct_url(self, airport: str, search_text: str, page: int = None) -> str:
        base_path = self.config.search_path.format(airport=airport)
        url = urljoin(self.config.base_url, base_path)
        params = {self.config.query_param: search_text}
        if page is not None and page > 0:
            params["page"] = page
        return requests.Request("GET", url, params=params).prepare().url

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            self.logger.debug(f"Requesting: {url}")
            response = self.session.get(url, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            time.sleep(self.config.rate_limit_delay)
            
            if not response.content:
                self.logger.warning("Empty response received")
                return None
                
            return BeautifulSoup(response.content, "html.parser")
        except RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching page: {str(e)}")
            return None

    def _parse_results_count(self, soup: BeautifulSoup) -> int:
        try:
            element = soup.select_one(self.selectors.results_count)
            if not element:
                return 0
                
            count_text = element.text.strip()
            count_match = re.search(r'(\d+)', count_text)
            if count_match:
                return int(count_match.group(1))
            return 0
        except Exception as e:
            self.logger.error(f"Failed to parse results count: {str(e)}")
            return 0

    def _string_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _passes_filters(self, product_data: ProductData, search_text: str, category: str = None) -> bool:
        try:
            def normalize(text):
                if not text:
                    return ""
                return re.sub(r'[^\w\s]', ' ', text.lower()).strip()

            search_query = normalize(search_text)
            product_name = normalize(product_data.name)
            
            if not product_name:
                self.logger.info("Rejected: Empty product name")
                return False
            
            required_keywords = [kw for kw in search_query.split() if kw]
            
            if category and normalize(product_data.category) != normalize(category):
                self.logger.info(f"Rejected: Category mismatch - expected '{category}', got '{product_data.category}'")
                return False

            for kw in required_keywords:
                name_parts = product_name.split()
                direct_match = any(kw in part for part in name_parts)
                
                similarity_match = any(self._string_similarity(kw, part) >= 0.8 for part in name_parts)
                
                whole_name_match = self._string_similarity(kw, product_name) >= 0.7
                
                if not (direct_match or similarity_match or whole_name_match):
                    self.logger.info(f"Rejected: Missing keyword '{kw}' in '{product_name}'")
                    return False

            self.logger.info(f"Approved: Product '{product_data.name}' passed all filters")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in filter processing: {str(e)}")
            return False

    def _extract_product_data(self, item: BeautifulSoup, airport: str) -> Optional[ProductData]:
        """Extract product data from HTML item."""
        self.current_product_count += 1
        product_log_prefix = f"Product #{self.current_product_count}"
        
        try:
            self.logger.debug(f"{product_log_prefix}: Processing product item")
            
            # Extract JSON data
            script = item.select_one(self.selectors.json_script)
            if not script:
                self.logger.debug(f"{product_log_prefix}: No JSON script found")
                return None

            try:
                json_data = json.loads(script.text.strip())
                if not json_data.get("id") or not json_data.get("name"):
                    self.logger.debug(f"{product_log_prefix}: Missing essential data in JSON")
                    return None
            except json.JSONDecodeError as e:
                self.logger.debug(f"{product_log_prefix}: JSON decode error: {str(e)}")
                return None

            product_link = item.select_one(self.selectors.product_link)
            if not product_link or 'href' not in product_link.attrs:
                self.logger.debug(f"{product_log_prefix}: No product link found")
                return None
            
            product_url = urljoin(self.config.base_url, product_link["href"])
            
            image = item.select_one(self.selectors.image)
            image_url = image.get("src") if image else None
            
            brand_element = item.select_one(self.selectors.brand)
            brand = brand_element.text.strip() if brand_element else json_data.get("brand", "")
            
            product_data = {
                "id": str(json_data.get("id", "")),
                "name": json_data.get("name", ""),
                "price": json_data.get("price", ""),
                "currency": "EUR",
                "brand": brand,
                "category": json_data.get("category", ""),
                "variant": json_data.get("dimension3", ""),
                "image_url": image_url,
                "product_url": product_url,
                "airport": airport,
            }

            return ProductData(**product_data)
            
        except (ValidationError, KeyError) as e:
            self.logger.debug(f"{product_log_prefix}: Data validation error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"{product_log_prefix}: Unexpected error: {str(e)}")
            return None

    def _scrape_product_details(self, product_url: str) -> Dict[str, Any]:
        details = {}
        try:
            self.logger.debug(f"Scraping details from: {product_url}")
            
            soup = self._get_page(product_url)
            if not soup:
                return details
            
            description_elements = soup.select(self.selectors.product_description)
            if description_elements:
                details["description"] = " ".join(elem.text.strip() for elem in description_elements)
            
            spec_table = soup.select_one(self.selectors.product_details_table)
            if spec_table:
                specifications = {}
                rows = spec_table.select(self.selectors.detail_row)
                for row in rows:
                    key_elem = row.select_one(self.selectors.detail_key)
                    val_elem = row.select_one(self.selectors.detail_value)
                    
                    if key_elem and val_elem:
                        key = key_elem.text.strip()
                        value = val_elem.text.strip()
                        
                        if key and value:
                            specifications[key] = value
                            
                            # Extract specific fields
                            if "item number" in key.lower():
                                details["item_number"] = value
                            elif "alcohol" in key.lower() and "%" in value:
                                try:
                                    abv_match = re.search(r'(\d+[.,]?\d*)\s*%', value)
                                    if abv_match:
                                        details["alcohol_by_volume"] = float(abv_match.group(1).replace(',', '.'))
                                except ValueError:
                                    pass
                            elif "manufacturer" in key.lower():
                                details["manufacturer_info"] = value
                            elif "warning" in key.lower():
                                details["warnings"] = value
                
                if specifications:
                    details["specifications"] = specifications
            
            if self.openai_client and soup:
                accordion_div = soup.find('div', class_='c-accordion js-accordion')
                if accordion_div:
                    ai_data = self._extract_data_with_ai(accordion_div)
                    if ai_data:
                        details["ai_extracted_data"] = ai_data
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error scraping product details: {str(e)}")
            return details

    def _extract_data_with_ai(self, html_element: BeautifulSoup) -> Optional[Dict[str, Any]]:
        if not self.openai_client:
            return None
            
        try:
            # Convert HTML to string, limiting to reasonable size
            html_str = str(html_element)
            if len(html_str) > 4000:  # Truncate if too large
                html_str = html_str[:4000] + "..."
            
            prompt = f"""
            Extract product details from this HTML in JSON format.
            Focus on: description, specifications, alcohol content, warnings, etc.
            Return VALID JSON only, no commentary or markdown.
            
            HTML Content:
            {html_str}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            json_start = content.find('{')
            json_end = content.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse AI-generated JSON")
                    return None
            else:
                self.logger.warning("No JSON found in AI response")
                return None
                
        except Exception as e:
            self.logger.error(f"AI extraction error: {str(e)}")
            return None

    def scrape(
        self,
        search_text: str,
        airport: str = None,
        max_products: Optional[int] = None,
        category: Optional[str] = None
    ) -> Tuple[int, List[ProductData]]:
        
        if not search_text:
            self.logger.error("Search text cannot be empty")
            return 0, []
            
        airport = airport or self.config.default_airport
        self.logger.info(f"Starting scrape for '{search_text}' at {airport} airport")
        
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
                self.logger.warning(f"Failed to retrieve page {page + 1}")
                break

            product_list = soup.select(self.selectors.product_list)
            if not product_list:
                self.logger.info(f"No product list found on page {page + 1}")
                break
                
            page_products = []
            skipped_count = 0
            consecutive_skips = 0

            for list_idx, product_list_element in enumerate(product_list, 1):
                items = product_list_element.select(self.selectors.product_item)
                self.logger.info(f"Found {len(items)} product items in list {list_idx}")
                
                for item in items:
                    if max_products and len(products) >= max_products:
                        self.logger.info(f"Reached maximum product count ({max_products})")
                        should_continue = False
                        break

                    product_data = self._extract_product_data(item, airport)
                    if not product_data:
                        skipped_count += 1
                        consecutive_skips += 1
                        continue

                    if self._passes_filters(product_data, search_text, category):
                        consecutive_skips = 0
                        details = self._scrape_product_details(product_data.product_url)
                        
                        updated_product = product_data.copy(update=details)
                        page_products.append(updated_product)
                        self.logger.info(f"Added product: {updated_product.name}")
                    else:
                        skipped_count += 1
                        consecutive_skips += 1

                    if consecutive_skips >= 10:
                        self.logger.info("Stopping due to 10 consecutive skips")
                        should_continue = False
                        break

                if not should_continue or (max_products and len(products) >= max_products):
                    break

            products.extend(page_products)
            total_pages_processed += 1
            
            self.logger.info(f"Page {page + 1} results: {len(page_products)} products added, {skipped_count} skipped")

            if len(page_products) > 0:
                page += 1
                consecutive_empty_pages = 0
            else:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 1:
                    self.logger.info("Stopping due to empty page")
                    should_continue = False

            if page_products and skipped_count > len(page_products) * 3:
                self.logger.info("Stopping due to high skip ratio")
                should_continue = False

            if max_products and len(products) >= max_products:
                should_continue = False

        self.logger.info(f"Scrape complete: {len(products)} products from {total_pages_processed} pages")
        return len(products), products
    
    def write_to_csv(self, products: List[ProductData], filename: str = None):
        if not products:
            self.logger.warning("No products to write to CSV")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            filename = f"products_{timestamp}.csv"
            
        filepath = os.path.join(self.config.output_dir, filename)
        
        try:
            sample_data = products[0].to_dict()
            fieldnames = list(sample_data.keys())
            
            with open(filepath, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                
                for product in products:
                    writer.writerow(product.to_dict())
                    
            self.logger.info(f"Wrote {len(products)} products to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to write CSV: {str(e)}")
            return None
    
    def write_to_json(self, products: List[ProductData], filename: str = None):
        if not products:
            self.logger.warning("No products to write to JSON")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            filename = f"products_{timestamp}.json"
            
        filepath = os.path.join(self.config.output_dir, filename)
        
        try:
            products_data = [product.to_dict() for product in products]
            
            with open(filepath, mode='w', encoding='utf-8') as file:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "count": len(products),
                    "products": products_data
                }, file, ensure_ascii=False, indent=2)
                    
            self.logger.info(f"Wrote {len(products)} products to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to write JSON: {str(e)}")
            return None


if __name__ == "__main__":
    config = ScraperConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),  
        log_level=logging.INFO,
        output_dir="output"
    )
    
    scraper = HeinemannScraper(config=config)
    
    results_count, products = scraper.scrape(
        search_text="johnnie walker blue", 
        category="whisky",
        max_products=100
    )
    
    scraper.write_to_csv(products)
    scraper.write_to_json(products)