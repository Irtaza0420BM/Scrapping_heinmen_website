# test.py
import logging
from pprint import pformat
from scrapper import HeinemannScraper, ScraperConfig, ProductData

def print_product(product: ProductData, number: int):
    """Print full product details in a human-readable format"""
    print(f"\n{'=' * 50}")
    print(f"PRODUCT #{number} DETAILS")
    print("=" * 50)
    fields = [
        ("ID", product.id),
        ("Name", product.name),
        ("Brand", product.brand),
        ("Price", f"{product.price} {product.currency}"),
        ("Category", product.category),
        ("Variant", product.variant),
        ("Item Number", product.item_number),
        ("Alcohol %", product.alcohol_by_volume),
        ("Manufacturer", product.manufacturer_info),
        ("Warnings", product.warnings),
        ("Description", (product.description or "")[:200] + "..."),
        ("Image URL", product.image_url),
        ("Product URL", product.product_url),
        ("Specifications", pformat(product.specifications))
    ]
    
    for label, value in fields:
        print(f"{label:15}: {value or 'N/A'}")
    
    print("=" * 50 + "\n")

def main():
    # Configure verbose logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Initialize scraper with aggressive rate limiting for debugging
    config = ScraperConfig(
        rate_limit_delay=2.0,
        max_retries=5,
        request_timeout=60
    )
    scraper = HeinemannScraper(config=config)
    
    # Scrape process with immediate output
    try:
        logging.info("🚀 Starting test scrape session")
        total_results, products = scraper.scrape(
            search_text="jonny blue walker",
            airport="fra",
            max_products=3  # Only process 3 products
        )
        
        logging.info(f"📊 Total results reported: {total_results}")
        logging.info(f"✅ Successfully processed {len(products)} products")
        
        # Print detailed results
        for idx, product in enumerate(products, 1):
            print_product(product, idx)
            
            # Pause between products for better readability
            if idx < len(products):
                input("Press Enter to view next product...")
                
    except Exception as e:
        logging.error(f"🔥 Critical error: {str(e)}")
    finally:
        logging.info("🏁 Test session completed")

if __name__ == "__main__":
    main()