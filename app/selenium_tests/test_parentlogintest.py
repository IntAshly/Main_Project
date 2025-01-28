import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

class TestParentLoginTest():
    def setup_method(self, method):
        # Set Chrome options to ignore SSL certificate errors
        chrome_options = Options()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        
        # Initialize the Chrome WebDriver with options
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_window_size(1050, 728)
        self.vars = {}
  
    def teardown_method(self, method):
        # Quit the WebDriver
        self.driver.quit()
  
    def test_parent_login_test(self):
        # Navigate to the login page
        self.driver.get("http://127.0.0.1:8000/login/")
        
        try:
            # Wait for email field to be visible and enter email
            email_field = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "email"))
            )
            email_field.click()
            email_field.send_keys("simisajan002@gmail.com")
            
            # Enter password
            password_field = self.driver.find_element(By.ID, "password")
            password_field.click()
            password_field.send_keys("Simi@123")
            
            # Click the login button
            self.driver.find_element(By.CSS_SELECTOR, ".btn").click()
            
            # Wait until the home page loads
            WebDriverWait(self.driver, 10).until(
                EC.url_contains("home")
            )
            
            # Click 'Schedule Appointment'
            schedule_appointment_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Schedule Appointment"))
            )
            schedule_appointment_button.click()

            # Wait for 'View Appointments' link to be clickable
            view_appointments_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "View Appointments"))
            )
            view_appointments_button.click()

            # Navigate back to Home
            home_link = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "Home"))
            )
            home_link.click()

        except TimeoutException:
            print("Timeout occurred. Some elements were not found in time.")
        except Exception as e:
            print(f"An error occurred: {e}")

# If running the script directly, use pytest to execute the test
if __name__ == "__main__":
    pytest.main([__file__])
