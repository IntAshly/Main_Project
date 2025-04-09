# test_add_to_cart.py

import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

class TestAddToCart:
    def setup_method(self, method):
        self.driver = webdriver.Chrome()  # Ensure chromedriver is in your PATH
        self.vars = {}

    def teardown_method(self, method):
        self.driver.quit()

    def test_add_to_cart(self):
        self.driver.get("http://127.0.0.1:8000/login/")
        self.driver.set_window_size(1382, 744)
        
        # Login
        self.driver.find_element(By.ID, "email").send_keys("simisajan002@gmail.com")
        self.driver.find_element(By.ID, "password").send_keys("Simi@123")
        self.driver.find_element(By.CSS_SELECTOR, ".btn").click()

        # Navigate to Buy Developmental Toys
        time.sleep(2)  # Wait for login to complete and page to load
        self.driver.find_element(By.LINK_TEXT, "Buy Developmental Toys").click()

        # Select first product
        time.sleep(2)  # Wait for page to load
        self.driver.find_element(By.CSS_SELECTOR, ".col-lg-3:nth-child(1) .product-image").click()

        # Add to cart
        time.sleep(2)  # Wait for product page to load
        self.driver.find_element(By.CSS_SELECTOR, ".btn").click()

        # Optionally assert some success message or redirection
        time.sleep(2)
        assert "cart" in self.driver.current_url.lower() or "added" in self.driver.page_source.lower()

# Run the test directly
if __name__ == "__main__":
    pytest.main(["-v", "test_addtocart.py"])
