




""" Classes definition."""
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


class BrowserProfileManager:    
    def __init__(self, randomize_user_agent=False, cookie_rotation=False):
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
