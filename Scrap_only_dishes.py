from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import csv


chromedriver_path = "chromedriver.exe"  
restaurant_links = [
    'https://www.zomato.com/visakhapatnam/subbayya-gari-hotel-dondaparithy-vizag/order',
    'https://www.zomato.com/kanpur/pa-ji-family-restro-rai-purwa/order',
    'https://www.zomato.com/kanpur/bikanervala-kakadeo/order',
    'https://www.zomato.com/kanpur/rominus-pizza-burger-mall-road/order',
    'https://www.zomato.com/kanpur/sidewalk-bakehouse-cafe-swaroop-nagar/info',
    'https://www.zomato.com/kanpur/waterside-the-landmark-hotel-mall-road/info',
    'https://www.zomato.com/kanpur/rajma-chawal-rajendra-nagar/order',
    'https://www.zomato.com/kanpur/oishi-the-temple-of-wok-parade/info', 
    'https://www.zomato.com/kanpur/studio-xo-bar-mall-road/info',
    'https://www.zomato.com/kanpur/zouk-air-bar-sharda-nagar/info',
    'https://www.zomato.com/kanpur/kake-di-hatti-saket-nagar/info'
]


options = webdriver.ChromeOptions()
options.add_argument("start-maximized")


service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service, options=options)


data = []

for url in restaurant_links:
    driver.get(url)
    time.sleep(5)  

    try:
        restaurant_name_elem = driver.find_element(By.TAG_NAME, 'h1')
        restaurant_name = restaurant_name_elem.text.strip()
    except:
        restaurant_name = "Unknown Restaurant"

    name_elements = driver.find_elements(By.XPATH, '//h4')
    price_elements = driver.find_elements(By.XPATH, '//span[contains(text(),"₹")]')

    for name_elem, price_elem in zip(name_elements, price_elements):
        name = name_elem.text.strip()
        price = price_elem.text.strip()
        if name and price.startswith("₹"):
            data.append([restaurant_name, name, price])


csv_filename = "zomato_menu_data.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Restaurant', 'Dish Name', 'Price'])
    writer.writerows(data)

print(f"\n Scraping complete. Data saved to '{csv_filename}'.")


driver.quit()