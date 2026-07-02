"""One-off helper: screenshot the running Streamlit app for the README.

Uses the system Chrome (channel='chrome') so no browser download is needed.
Run with the app already serving on http://localhost:8501:
    python docs/_shots.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def wait_ready(page):
    # Wait until Streamlit has computed and rendered the ₹ prediction metric.
    page.wait_for_selector('[data-testid="stMetricValue"]', timeout=30000)
    page.wait_for_function(
        "() => (document.querySelector('[data-testid=\"stMetricValue\"]')?.innerText || '').includes('₹')",
        timeout=30000,
    )
    page.wait_for_timeout(1200)  # let fonts/layout settle


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1000},
                                device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        wait_ready(page)
        page.screenshot(path=str(OUT / "app_home.png"), full_page=True)
        print("saved app_home.png")

        browser.close()


if __name__ == "__main__":
    main()
