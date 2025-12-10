#!/usr/bin/env python3
"""
Convert HTML file to PNG image for LaTeX inclusion.
Requires: pip install playwright
Then run: playwright install chromium
"""

import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Install with: pip install playwright")
    print("Then run: playwright install chromium")
    sys.exit(1)

def html_to_image(html_path, output_path, width=1200, height=None):
    """Convert HTML file to PNG image."""
    html_path = Path(html_path)
    output_path = Path(output_path)
    
    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        return False
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height or 800})
        
        # Load the HTML file
        page.goto(f"file://{html_path.absolute()}")
        
        # Wait for content to load
        page.wait_for_timeout(2000)  # Wait 2 seconds for any dynamic content
        
        # Take screenshot
        page.screenshot(path=str(output_path), full_page=True)
        
        browser.close()
    
    print(f"Successfully created image: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_html_to_image.py <html_file> [output_image]")
        print("Example: python convert_html_to_image.py ui.html ui_screenshot.png")
        sys.exit(1)
    
    html_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "ui_screenshot.png"
    
    html_to_image(html_file, output_file, width=1400, height=None)

