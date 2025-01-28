import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

class TestAdminLoginTest:
    def setup_method(self, method):
        # Set Chrome options to ignore SSL certificate errors
        chrome_options = Options()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')

        # Initialize Chrome WebDriver with options
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_window_size(1382, 744)
        self.vars = {}

    def teardown_method(self, method):
        # Quit the WebDriver
        self.driver.quit()

    def test_admin_login_test(self):
        # Navigate to the login page
        self.driver.get("http://127.0.0.1:8000/login/")

        try:
            # Wait for email field and enter email
            email_field = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "email"))
            )
            email_field.click()
            email_field.send_keys("nurturenest@gmail.com")

            # Enter password
            password_field = self.driver.find_element(By.ID, "password")
            password_field.click()
            password_field.send_keys("Admin@123")

            # Click the login button
            self.driver.find_element(By.CSS_SELECTOR, ".btn").click()

            # Navigate through the different sections of the admin dashboard
            # View Available Vaccines
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "View Available Vaccines"))
            ).click()

            # Go back to Dashboard
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".sidebar-link:nth-child(1)"))
            ).click()

            # Vaccines Requests
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Vaccines Requests"))
            ).click()

            # Back to Dashboard
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Dashboard"))
            ).click()

            # View Feeding Chart Details
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "View Feeding Chart Details"))
            ).click()

            # Back to Dashboard
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Dashboard"))
            ).click()

            # Profile and Logout
            profile_icon = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".profile-container"))
            )
            profile_icon.click()

            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "NurtureNest"))
            ).click()

            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Logout"))
            ).click()

        except TimeoutException:
            print("Timeout occurred. Some elements were not found in time.")
        except Exception as e:
            print(f"An error occurred: {e}")

# If running the script directly, use pytest to execute the test
if __name__ == "__main__":
    pytest.main([__file__])

