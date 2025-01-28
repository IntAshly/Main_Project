from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Function to perform login test based on user role and validate home page navigation
def test_login(driver, role, email, password, expected_page_url, should_succeed=True):
    # Open the login page in a new tab
    driver.execute_script("window.open('http://127.0.0.1:8000/login/', '_blank');")  # Replace with your actual login URL
    driver.switch_to.window(driver.window_handles[-1])  # Switch to the new tab

    try:
        # Wait for the email input to be visible and enter the email
        email_input = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "email"))  # Adjust the selector as needed
        )
        time.sleep(1)  # Wait for 1 second before entering the email
        email_input.send_keys(email)

        # Wait for the password input to be visible and enter the password
        password_input = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "password"))  # Adjust the selector as needed
        )
        time.sleep(1)  # Wait for 1 second before entering the password
        password_input.send_keys(password)

        # Submit the form
        password_input.send_keys(Keys.RETURN)

        if should_succeed:
            # Wait for the expected page URL to load
            WebDriverWait(driver, 10).until(
                EC.url_contains(expected_page_url)  # Wait until the expected page URL is loaded
            )
            print(f"Navigation to {expected_page_url} successful for {role}.")

            # Logout from the dashboard
            logout_button = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, "logout"))  # Adjust the selector as needed
            )
            logout_button.click()

            # Wait for the login page to load again
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.NAME, "email"))  # Adjust the selector as needed
            )
        else:
            # For a failed login, check if an error message is displayed on the page
            error_message = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CLASS_NAME, "error-message"))  # Adjust the selector for error message
            )
            print(f"Login failed for {role} with email {email}. Error message: {error_message.text}")
            time.sleep(2)  # Wait to display the error message

            # Stop the script if login fails
            driver.quit()
            print("Program execution stopped after detecting login failure.")
            return

    except Exception as e:
        print(f"An error occurred during {role} login: {e}")
    finally:
        driver.close()  # Close the current tab
        driver.switch_to.window(driver.window_handles[0])  # Switch back to the original tab

# Set up the WebDriver (make sure you have the appropriate driver installed)
driver = webdriver.Edge()  # or use `webdriver.Chrome()` depending on your browser

# Run the login test for different users and check for successful navigation
test_login(driver, "parent", "simisajan002@gmail.com", "Simi@123", "home.html")  # Test parent login and navigate to home.html
test_login(driver, "admin", "nurturenest@gmail.com", "Admin@123", "admin_home.html")  # Test admin login and navigate to admin_home.html
test_login(driver, "health center", "medtech@gmail.com", "Medtech@123", "health_home.html")  # Test health center login and navigate to health_home.html

# Add the test case for the user where login should fail and display an error message
test_login(driver, "parent", "simi@gmail.com", "Simi@002", "home.html", should_succeed=False)  # Test failed login for this user


# Close the browser
driver.quit()
