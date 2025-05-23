# Generated by Selenium IDE
import pytest
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

class TestTest6():
  def setup_method(self, method):
    self.driver = webdriver.Chrome()
    self.vars = {}
  
  def teardown_method(self, method):
    self.driver.quit()
  
  def test_test6(self):
    self.driver.get("http://127.0.0.1:8000/login/")
    self.driver.set_window_size(1382, 744)
    self.driver.find_element(By.ID, "email").click()
    self.driver.find_element(By.ID, "email").send_keys("simisajan002@gmail.com")
    self.driver.find_element(By.ID, "password").click()
    self.driver.find_element(By.ID, "password").send_keys("Simi@123")
    self.driver.find_element(By.ID, "password").send_keys(Keys.ENTER)
    self.driver.find_element(By.LINK_TEXT, "Buy Developmental Toys").click()
    self.driver.find_element(By.CSS_SELECTOR, ".col-lg-3:nth-child(1) .text-dark").click()
  
# Run the test directly
if __name__ == "__main__":
    pytest.main(["-v", "test_test6.py"])