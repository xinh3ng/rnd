from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

# Setup webdriver
webdriver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=webdriver_service)

# Navigate to the webpage
driver.get("https://www.youtube.com")

# Let the JavaScript load
sleep(5)  # Adjust this delay as needed

# Find all video link elements on the page
video_elements = driver.find_elements("xpath", '//*[@id="video-title"]')

# Extract and print out the URLs of the videos
for video_element in video_elements:
    print(video_element.get_attribute("href"))

# Close the browser
driver.quit()
