from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import random

app = Flask(__name__)

def scrape_1mg(medicine_name):
    # Setup Chrome driver
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    chrome_driver_path = r"C:\Users\dhana\Downloads\chromedriver-win64\chromedriver.exe"
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)
    driver.set_page_load_timeout(30)

    result = {}
    search_url = f"https://www.1mg.com/search/all?name={medicine_name}"

    try:
        driver.get(search_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/drugs/'], a[href^='/otc/']"))
        )
        time.sleep(random.uniform(2, 3))
        items = driver.find_elements(By.CSS_SELECTOR, "a[href^='/drugs/'], a[href^='/otc/']")
        href = None
        for item in items:
            link = item.get_attribute("href")
            if link:
                href = link
                break
        if not href:
            result["error"] = "No product found."
            return result

        driver.get(href)
        time.sleep(random.uniform(2, 3))

        result["Medicine Name"] = medicine_name
        result["Brand Name"] = driver.find_element(By.CLASS_NAME, "DrugHeader__title-content___2ZaPo").text if driver.find_elements(By.CLASS_NAME, "DrugHeader__title-content___2ZaPo") else "N/A"
        result["Manufacturer"] = driver.find_element(By.XPATH, "//div[contains(text(),'Marketer')]/following-sibling::div").text if driver.find_elements(By.XPATH, "//div[contains(text(),'Marketer')]/following-sibling::div") else "N/A"
        result["Salt"] = driver.find_element(By.XPATH, "//div[contains(text(),'SALT COMPOSITION')]/following-sibling::div").text if driver.find_elements(By.XPATH, "//div[contains(text(),'SALT COMPOSITION')]/following-sibling::div") else "N/A"
        result["Price"] = driver.find_element(By.CSS_SELECTOR, "span.PriceBoxPlanOption__offer-price___3v9x8").text if driver.find_elements(By.CSS_SELECTOR, "span.PriceBoxPlanOption__offer-price___3v9x8") else "N/A"

        uses = driver.find_elements(By.CSS_SELECTOR, "#uses_and_benefits li")
        result["Uses"] = ', '.join([li.text for li in uses]) if uses else "N/A"

        side_effects = driver.find_elements(By.CSS_SELECTOR, "#side_effects ul li")
        result["Side Effects"] = ', '.join([li.text for li in side_effects]) if side_effects else "N/A"

        result["URL"] = href

    except (TimeoutException, WebDriverException) as e:
        result["error"] = str(e)
    finally:
        driver.quit()

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    data = None
    if request.method == 'POST':
        med_name = request.form['medicine']
        data = scrape_1mg(med_name)
    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
