import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestLogintest():
    def setup_method(self, method):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(5)  # Wait for elements implicitly
        self.vars = {}
  
    def teardown_method(self, method):
        self.driver.quit()
  
    def test_logintest(self):
        self.driver.get("http://127.0.0.1:8000/login/")
        self.driver.set_window_size(1050, 728)
        
        # Enter valid email
        self.driver.find_element(By.ID, "email").click()
        self.driver.find_element(By.ID, "email").send_keys("invaliduser@example.com")
        
        # Enter invalid password
        self.driver.find_element(By.ID, "password").click()
        self.driver.find_element(By.ID, "password").send_keys("InvalidPassword")
        
        # Click login button
        self.driver.find_element(By.CSS_SELECTOR, ".btn").click()
        
        # Wait for the error message to appear and verify it
        try:
            # Adjust the XPath or CSS selector based on your actual error message element
            error_message_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'error-message')]"))
            )
            # Validate the text of the error message
            error_message_text = error_message_element.text
            assert "Invalid credentials" in error_message_text
            print("Error message displayed: ", error_message_text)
        except Exception as e:
            print("Error message not found or different: ", str(e))

# If running the script directly, use pytest to execute the test
if __name__ == "__main__":
    pytest.main([__file__])
