from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest

class TestViewProducts:
    def setup_method(self):
        # Initialize WebDriver (Ensure you have the correct path for your ChromeDriver)
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        self.driver.maximize_window()

    def teardown_method(self):
        # Close the WebDriver after test completion
        self.driver.quit()

    def test_view_products(self):
        self.driver.get("http://127.0.0.1:8000/home/")

        # Wait for 'Buy Developmental Toys' link to be clickable and click it
        buy_toys_link = self.wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Buy Developmental Toys")))
        buy_toys_link.click()

        # Wait for the first product to be visible and click it
        first_product = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".col-lg-3:nth-child(1) .text-dark")))
        first_product.click()

        # Wait for the "Add to Cart" or "Buy" button to be clickable
        buy_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-outline-primary")))
        self.driver.execute_script("arguments[0].scrollIntoView();", buy_button)  # Scroll into view if needed
        buy_button.click()

if __name__ == "__main__":
    pytest.main()
