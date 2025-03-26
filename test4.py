import json
import logging
import time
import random
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Set
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
from dotenv import load_dotenv
from datetime import datetime
from fake_useragent import UserAgent
import socket
from functools import partial


class ScraperConfig(BaseModel):
    base_url: str = "https://www.heinemann-shop.com"
    default_airport: str = "fra"
    search_path: str = "/en/{airport}/search/"
    query_param: str = "text"
    request_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    rate_limit_delay: float = 1.0
    rate_limit_jitter: float = 0.5  
    randomize_user_agent: bool = True
    proxy_list: List[str] = Field(default_factory=list) 
    headers: Dict[str, str] = Field(default_factory=lambda: {
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.heinemann-shop.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    api_key: Optional[str] = None
    log_level: int = logging.INFO
    output_dir: str = "output"
    max_workers: int = 5  
    cookie_rotation: bool = True  
    session_rotation_count: int = 20 


class Selectors(BaseModel):
    """CSS selectors for scraping Heinemann shop."""
    results_count: str = "h1.m-category-overview__title"
    product_list: str = "ul.c-article-list"
    product_item: str = "li.c-article-list__item"
    json_script: str = "script[type='application/json']"
    brand: str = "span[itemprop='brand']"
    price: str = "div.c-article-tile__price"
    image: str = "img.c-article-list-item__image"
    product_link: str = "a.js-ga-article-link"
    accordion: str = "div.c-accordion.js-accordion"  

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
    ai_extracted_data: Optional[Dict[str, Any]] = None
    scrape_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProductData instance to dictionary with proper serialization."""
        data = self.model_dump(
            exclude_none=True,
            exclude={"specifications"}  # Exclude raw specifications if needed
        )
        
        # Handle special fields
        if hasattr(self, "specifications"):
            data["specifications"] = json.dumps(self.specifications)
            
        if self.ai_extracted_data:
            data["ai_extracted_data"] = json.dumps(self.ai_extracted_data)
            
        return data
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    class Config:
        extra = "allow"  

    @validator('name', 'brand', 'category', 'variant', pre=True)
    def clean_text_fields(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = re.sub(r'\s+', ' ', v).strip()
            cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
            return cleaned
        return v
    
    @validator('price', pre=True)
    def clean_price(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            price_match = re.search(r'(\d+[.,]?\d*)', v)
            if price_match:
                return price_match.group(1).replace(',', '.')
        return v
    
class BrowserProfileManager:
    
    def __init__(self, randomize_user_agent=True, cookie_rotation=True):
        self.randomize_user_agent = randomize_user_agent
        self.cookie_rotation = cookie_rotation
        self.user_agent_generator = UserAgent()
        self.proxy_index = 0
        self.request_count = 0
        
        self.screen_resolutions = [
            "1920x1080", "1366x768", "1536x864", "1440x900", 
            "1280x720", "1600x900", "1024x768", "2560x1440"
        ]
        
        self.platforms = [
            "Windows NT 10.0; Win64; x64", 
            "Macintosh; Intel Mac OS X 10_15_7",
            "X11; Linux x86_64",
            "Windows NT 6.1; Win64; x64",
            "Macintosh; Intel Mac OS X 10_14_6"
        ]
        
        self.cookie_jars = [requests.cookies.RequestsCookieJar() for _ in range(5)]
    
    def get_random_user_agent(self) -> str:
        if self.randomize_user_agent:
            try:
                return self.user_agent_generator.random
            except Exception:
                return (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/94.0.4606.81 Safari/537.36"
                )
        else:
            return (
                "Mozilla/5.0 (compatible; ScraperBot/1.0; +https://example.com/scraper)"
            )
    
    def get_random_headers(self) -> Dict[str, str]:
        user_agent = self.get_random_user_agent()
        platform = random.choice(self.platforms)
        resolution = random.choice(self.screen_resolutions)
        
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": f"en-US,en;q=0.{random.randint(5, 9)}",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Referer": "https://www.google.com/",
            "Cache-Control": "max-age=0"
        }
        
        if random.random() > 0.5:
            headers["Viewport-Width"] = resolution.split('x')[0]
            headers["Device-Memory"] = f"{2 ** random.randint(1, 4)}"
        
        return headers
    
    def get_cookies(self):
        if self.cookie_rotation:
            jar_index = (self.request_count // 20) % len(self.cookie_jars)
            return self.cookie_jars[jar_index]
        return None
    
    def increment_request_count(self):
        self.request_count += 1
    
    def should_rotate_session(self) -> bool:
        if not self.cookie_rotation:
            return False
        return self.request_count > 0 and self.request_count % 20 == 0
    
    def get_next_proxy(self, proxy_list: List[str]) -> Optional[str]:
        if not proxy_list:
            return None
        
        proxy = proxy_list[self.proxy_index % len(proxy_list)]
        self.proxy_index += 1
        return proxy

class HeinemannScraper:
    """Scraper for Heinemann shop product data."""
    
    def __init__(self, config: Optional[ScraperConfig] = None, selectors: Optional[Selectors] = None):
        """Initialize the scraper with configuration and selectors."""
        self.config = config or ScraperConfig()
        self.selectors = selectors or Selectors()
        self.browser_manager = BrowserProfileManager(
            randomize_user_agent=self.config.randomize_user_agent,
            cookie_rotation=self.config.cookie_rotation
        )
        self.session = self._configure_session()
        self.logger = self._configure_logger()
        self.current_product_count = 0
        self.openai_client = self._configure_openai()
        self.request_count = 0
        
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        try:
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir, exist_ok=True)
                self.logger.info(f"Created output directory: {self.config.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {str(e)}")
            self.config.output_dir = "."

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
        
        # Set initial headers
        headers = self.browser_manager.get_random_headers()
        session.headers.update(headers)
        
        return session

    def _configure_logger(self) -> logging.Logger:
        """Configure logger with appropriate handlers and formatters."""
        logger = logging.getLogger("heinemann_scraper")
        
        # Clear any existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
            
        logger.setLevel(self.config.log_level)
        
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(os.path.join(self.config.output_dir, "scraper.log"))
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler(
                os.path.join(self.config.output_dir, "scraper.log"),
                mode='a'  # Append mode
            )
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # If file handler fails, just use console handler
            print(f"Error setting up file logger: {str(e)}")
            if not logger.handlers:
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
        
        return logger

    def _configure_openai(self) -> Optional[OpenAI]:
        """Configure OpenAI client if API key is available."""
        if self.config.api_key:
            try:
                return OpenAI(api_key=self.config.api_key)
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                return None
        else:
            self.logger.warning("No OpenAI API key provided. AI extraction will be skipped.")
            return None

    def _rotate_session_if_needed(self):
        """Rotate session parameters if needed based on request count."""
        self.request_count += 1
        self.browser_manager.increment_request_count()
        
        if self.browser_manager.should_rotate_session():
            self.logger.debug("Rotating session parameters")
            
            # Update user agent and headers
            headers = self.browser_manager.get_random_headers()
            self.session.headers.update(headers)
            
            # Maybe change proxy
            if self.config.proxy_list:
                proxy = self.browser_manager.get_next_proxy(self.config.proxy_list)
                if proxy:
                    self.session.proxies = {
                        'http': proxy,
                        'https': proxy
                    }
                    self.logger.debug(f"Switched to proxy: {proxy}")

    def _add_request_delay(self):
        """Add a randomized delay between requests to avoid rate limiting."""
        base_delay = self.config.rate_limit_delay
        jitter = self.config.rate_limit_jitter
        
        # Calculate random delay with jitter
        delay = base_delay + random.uniform(-jitter, jitter)
        delay = max(0.1, delay)  # Ensure delay is at least 0.1 seconds
        
        time.sleep(delay)

    def _construct_url(self, airport: str, search_text: str, page: int = None) -> str:
        """Construct search URL with parameters."""
        base_path = self.config.search_path.format(airport=airport)
        url = urljoin(self.config.base_url, base_path)
        params = {self.config.query_param: search_text}
        if page is not None and page > 0:
            params["page"] = page
        return requests.Request("GET", url, params=params).prepare().url

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse page content with error handling."""
        try:
            self._rotate_session_if_needed()
            
            self.logger.debug(f"Requesting: {url}")
            
            # Get the appropriate cookie jar
            cookies = self.browser_manager.get_cookies()
            
            # Add a random delay
            self._add_request_delay()
            
            # Make the request
            response = self.session.get(
                url, 
                timeout=self.config.request_timeout,
                cookies=cookies,
                allow_redirects=True
            )
            
            response.raise_for_status()
            
            # Update cookies if we got any
            if cookies is not None and response.cookies:
                cookies.update(response.cookies)
            
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
        """Parse the number of results from search page."""
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
        """Calculate string similarity ratio using SequenceMatcher."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _passes_filters(self, product_data: ProductData, search_text: str, category: str = None) -> bool:
        """Check if product passes defined filters."""
        try:
            def normalize(text):
                if not text:
                    return ""
                # Replace special characters, remove punctuation, convert to lowercase
                return re.sub(r'[^\w\s]', ' ', text.lower()).strip()

            search_query = normalize(search_text)
            product_name = normalize(product_data.name)
            
            # Skip empty product names
            if not product_name:
                self.logger.info("Rejected: Empty product name")
                return False
            
            # Check required keywords
            required_keywords = [kw for kw in search_query.split() if kw]
            
            # 1. Category filter (optional)
            if category:
                product_category = normalize(product_data.category)
                if not product_category:
                    self.logger.info(f"Rejected: Empty category")
                    return False
                
                # Check for category match with some flexibility
                if not (product_category == normalize(category) or 
                        self._string_similarity(product_category, normalize(category)) >= 0.8):
                    self.logger.info(f"Rejected: Category mismatch - expected '{category}', got '{product_data.category}'")
                    return False

            # 2. Keyword validation
            for kw in required_keywords:
                # Direct match in name parts
                name_parts = product_name.split()
                direct_match = any(kw in part for part in name_parts)
                
                # Similarity check for typos/variations
                similarity_match = any(self._string_similarity(kw, part) >= 0.8 for part in name_parts)
                
                # Whole name similarity check
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
            
            # Extract image URL
            image = item.select_one(self.selectors.image)
            image_url = image.get("src") if image else None
            
            # Extract brand (fallback to JSON data)
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
        """Scrape detailed product information from product page."""
        details = {}
        try:
            self.logger.debug(f"Scraping details from: {product_url}")
            
            soup = self._get_page(product_url)
            if not soup:
                return details
            
            if self.openai_client and soup:
                accordion_div = soup.select_one(self.selectors.accordion)
                if accordion_div:
                    details = self._extract_ai_data_and_merge(accordion_div, details)
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error scraping product details: {str(e)}")
            return details
    def _extract_ai_data_and_merge(self, html_element: BeautifulSoup, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from HTML using OpenAI and merge it with existing data.
        
        Args:
            html_element: BeautifulSoup element containing HTML to process
            existing_data: Existing data dictionary to merge with AI-extracted fields
            
        Returns:
            Updated dictionary with AI-extracted fields merged directly
        """
        if not self.openai_client:
            return existing_data
            
        try:
            
            prompt = f"""
            you are an expert Scrapper your task is to find the data that can be scrapped.
            I want you to scrape product description, product details, Ingredients and Taste. They sometimes may not be available in that case call tha json null.
            If any of the data doesn't exist set it to null, Some of them are tables with further key value pair. 
            Make them a JSON. 
            Return VALID JSON only, no commentary or markdown.
            
            HTML Content:
            {html_element}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            content = content.replace('\n', ' ')  # Improve JSON parsing
            
            self.logger.debug(f"Raw AI response: {content}")
            
            json_str = self._extract_json(content)
            
            if json_str:
                ai_data = json.loads(json_str)
                # Handle nested structures
                return self._handle_nested_data(ai_data, existing_data)
                
        except Exception as e:
            self.logger.error(f"AI processing failed: {str(e)}")
        return existing_data

    def _extract_json(self, content: str) -> Optional[str]:
        """Robust JSON extraction with multiple fallbacks"""
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                return match.group()
            return None

    def _handle_nested_data(self, ai_data: dict, existing_data: dict) -> dict:
        """Process nested JSON structures"""
        # Flatten nested objects
        flattened = {}
        for key, value in ai_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            else:
                flattened[key] = value
        
        # Merge with existing data
        existing_data["ai_extracted_data"] = flattened
        return existing_data
        
    def _process_product_details_batch(self, products_batch: List[ProductData]) -> List[ProductData]:
        """Process a batch of products to get their details."""
        results = []
        
        def fetch_details(product):
            try:
                details = self._scrape_product_details(product.product_url)
                updated_product = product.model_copy(update=details)
                self.logger.info(f"Successfully scraped details for: {product.name}")
                return updated_product
            except Exception as e:
                self.logger.error(f"Error processing {product.name}: {str(e)}")
                return product
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_product = {executor.submit(fetch_details, product): product for product in products_batch}
            
            for future in concurrent.futures.as_completed(future_to_product):
                product = future_to_product[future]
                try:
                    updated_product = future.result()
                    results.append(updated_product)
                except Exception as e:
                    self.logger.error(f"Worker thread error for {product.name}: {str(e)}")
                    results.append(product)  
        
        return results

    def scrape(
        self,
        search_text: str,
        airport: str = None,
        max_products: Optional[int] = None,
        category: Optional[str] = None,
        batch_size: int = 5
    ) -> Tuple[int, List[ProductData]]:
        """
        Scrape products matching search criteria.
        
        Args:
            search_text: Text to search for
            airport: Airport code (defaults to config default)
            max_products: Maximum number of products to scrape
            category: Optional category filter
            batch_size: Size of batches for parallel processing
            
        Returns:
            Tuple of (count, products list)
        """
        # Validate inputs
        if not search_text:
            self.logger.error("Search text cannot be empty")
            return 0, []
            
        airport = airport or self.config.default_airport
        self.logger.info(f"Starting scrape for '{search_text}' at {airport} airport")
        
        # Reset counters
        self.current_product_count = 0
        self.request_count = 0
        
        candidate_products = []  # Products that need detail scraping
        products = []  # Final processed products
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
                        
                        updated_product = product_data.model_copy(update=details)
                        page_products.append(updated_product)
                        self.logger.info(f"Added product: {updated_product.name}")
                    else:
                        skipped_count += 1
                        consecutive_skips += 1

                    if consecutive_skips >= 40:
                        self.logger.info("Stopping due to 40 consecutive skips")
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
            fieldnames = list(ProductData.model_json_schema()["properties"].keys())
        
            with open(filepath, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
            
                for product in products:
                    writer.writerow(product.model_dump())
        except Exception as e:
            self.logger.error(f"CSV write error: {str(e)}")

    
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
    load_dotenv() 

    config = ScraperConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),  
        log_level=logging.INFO,
        output_dir="output"
    )
    
    scraper = HeinemannScraper(config=config)
    
    results_count, products = scraper.scrape(
        search_text="johnnie walker blue", 
        airport = "fra",
        category="whisky",
        max_products=100
    )


    scraper.write_to_csv(products)
    scraper.write_to_json(products)