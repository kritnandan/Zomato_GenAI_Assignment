import random
import time
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv


chromedriver_path = "chromedriver.exe"
restaurant_links = [
    'https://www.zomato.com/visakhapatnam/subbayya-gari-hotel-dondaparithy-vizag/order',
    'https://www.zomato.com/kanpur/pa-ji-family-restro-rai-purwa/order',
    'https://www.zomato.com/kanpur/bikanervala-kakadeo/order',
    'https://www.zomato.com/kanpur/rominus-pizza-burger-mall-road/order',
    'https://www.zomato.com/visakhapatnam/vijayawada-ruchulu-ram-nagar-vizag/order',
    'https://www.zomato.com/visakhapatnam/meesala-rajula-ruchulu-seethammadhara-vizag/order',
    'https://www.zomato.com/visakhapatnam/sivakotis-food-magic-vishalaksmi-nagar-vizag/order',
    'https://www.zomato.com/visakhapatnam/mee-vivaha-bhojanambu-tiffins-gajuwaka-vizag/order',
    'https://www.zomato.com/kanpur/sidewalk-bakehouse-cafe-swaroop-nagar/order',
    'https://www.zomato.com/kanpur/cafe-coffee-day-1-mall-road/order',
    'https://www.zomato.com/kanpur/bistro-57-cafe-mall-road/order'
]

ua = UserAgent()
user_agents = [ua.random for _ in range(5)]  

chrome_options = Options()
chrome_options.add_argument("--headless")  
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-software-rasterizer")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("lang=en")


capabilities = DesiredCapabilities.CHROME
capabilities['goog:loggingPrefs'] = {'performance': 'ALL'}


service = Service(chromedriver_path)
chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
driver = webdriver.Chrome(service=service, options=chrome_options)


data = []


def scrape_page(url):
    try:
       
        headers = {"User-Agent": random.choice(user_agents)}

        
        driver.get(url)
        
        
        action = ActionChains(driver)
        action.send_keys(Keys.PAGE_DOWN).perform()
        time.sleep(random.uniform(2, 4))
        action.send_keys(Keys.PAGE_DOWN).perform()
        time.sleep(random.uniform(3, 5))

        
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))

        
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        
        restaurant_name = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Not Found"
        address = soup.find('div', class_='sc-clNaTc')
        address = address.get_text(strip=True) if address else "Not Available"
        status = soup.find('span', class_='sc-iGPElx')
        status = status.get_text(strip=True) if status else "Unknown"
        timings = soup.find('span', class_='sc-kasBVs')
        timings = timings.get_text(strip=True) if timings else "Not Available"
        contact_tag = soup.find('a', href=lambda href: href and "tel:" in href)
        contact_number = contact_tag.get_text(strip=True) if contact_tag else "Not Listed"

        
        name_elements = driver.find_elements(By.XPATH, '//h4')
        price_elements = driver.find_elements(By.XPATH, '//span[contains(text(),"₹")]')

        if name_elements and price_elements:
            for name_elem, price_elem in zip(name_elements, price_elements):
                dish_name = name_elem.text.strip()
                price = price_elem.text.strip()
                if dish_name and price.startswith("₹"):
                    data.append({
                        'Restaurant': restaurant_name,
                        'Address': address,
                        'Status': status,
                        'Timings': timings,
                        'Contact': contact_number,
                        'Dish Name': dish_name,
                        'Price': price
                    })
        else:
            
            data.append({
                'Restaurant': restaurant_name,
                'Address': address,
                'Status': status,
                'Timings': timings,
                'Contact': contact_number,
                'Dish Name': "N/A",
                'Price': "N/A"
            })
    except Exception as e:
        print(f"[ERROR] Error while scraping {url}: {e}")


for url in restaurant_links:
    print(f"\n[INFO] Processing: {url}")
    scrape_page(url)
    time.sleep(random.uniform(3, 7))  


csv_filename = "zomato_full_data.csv"
fieldnames = ['Restaurant', 'Address', 'Status', 'Timings', 'Contact', 'Dish Name', 'Price']

with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"\n✅ Scraping complete. Data saved to '{csv_filename}'.")


driver.quit()
