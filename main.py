from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

# options = Options()
# options.headless = True
# options.add_argument("--window-size=1920,1200")

# DRIVER_PATH = 'C:/Users/CSUFTitan/Downloads/chromedriver_win32/chromedriver.exe'
# driver = webdriver.Chrome(executable_path=DRIVER_PATH)
# driver.get('https://wikipedia.com')
# search_language = driver.find_element_by_id('searchLanguage')
# var = search_language.text
# print(var)
# Select(search_language).select_by_visible_text('Français')
# search_textinput = driver.find_element_by_id('searchInput')
# search_textinput.send_keys('Hrithik Roshan')
# search_button = driver.find_element_by_xpath('/html/body/div[3]/form/fieldset/button')
# search_button.click()
# driver.quit()


import time
from csv import DictWriter
import pprint
import datetime
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def init_driver(driver_type):
    if driver_type == 1:
        driver = webdriver.Firefox()
    elif driver_type == 2:
        DRIVER_PATH = 'C:/Users/CSUFTitan/Downloads/chromedriver_win32/chromedriver.exe'
        driver = webdriver.Chrome(executable_path=DRIVER_PATH)
        #driver = webdriver.Chrome()
    elif driver_type == 3:
        driver = webdriver.Ie()
    elif driver_type == 4:
        driver = webdriver.Opera()
    elif driver_type == 5:
        driver = webdriver.PhantomJS()
    driver.wait = WebDriverWait(driver, 5)
    return driver


def scroll(driver, start_date, end_date, words, lang, max_time=180):
    languages = { 1: 'en', 2: 'it', 3: 'es', 4: 'fr', 5: 'de', 6: 'ru', 7: 'zh'}
    url = "https://twitter.com/search?q="
    for w in words[:-1]:
        url += "{}%20OR".format(w)
    url += "{}%20".format(words[-1])
    url += "since%3A{}%20until%3A{}&".format(start_date, end_date)
    if lang != 0:
        url += "l={}&".format(languages[lang])
    url += "src=typd"
    print(url)
    driver.get(url)
    start_time = time.time()  # remember when we started
    while (time.time() - start_time) < max_time:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        print( str(time.time() - start_time) + " < " + str(max_time) )


def scrape_tweets(driver):
    try:
        tweet_divs = driver.page_source
        obj = (tweet_divs, "html.parser")
        content = obj.find_all("div", class_="content")
        print(content)

        print("content printed")
        print(len(content))
        for c in content:
            tweets = c.find("p", class_="tweet-text").strings
            tweet_text = "".join(tweets)
            print(tweet_text)
            print("-----------")
            try:
                name = (c.find_all("strong", class_="fullname")[0].string).strip()
            except AttributeError:
                name = "Anonymous"
            date = (c.find_all("span", class_="_timestamp")[0].string).strip()

            datestring = str(c.find_all("span", class_="_timestamp")[0])
            print(datestring)
            datestring = datestring[datestring.index("data-time")+11:]
            datestring = datestring[:datestring.index("\"")]
            print(datestring)
            # print(tweet_text)
            try:
                write_csv(datestring,tweet_text,name)
            except:
                print('csv error')

    except Exception as e:
        print("Something went wrong!")
        print(e)
        driver.quit()


def write_csv_header():
    with open("twitterData.csv", "w+") as csv_file:
        fieldnames = ['Date', 'Name', 'Tweets','Tags']
        writer = DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def write_csv(date,tweet,name):
    with open("twitterData.csv", "a+") as csv_file:
        fieldnames = ['Date', 'Name', 'Tweets','Tags']
        writer = DictWriter(csv_file, fieldnames=fieldnames)
        #writer.writeheader()
        writer.writerow({'Date': date,'Name': name,'Tweets': tweet})



def make_csv(data):
    l = len(data['date'])
    print("count: %d" % l)
    with open("twitterData.csv", "a+") as file:
        fieldnames = ['Date', 'Name', 'Tweets']
        writer = DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(l):
            writer.writerow({'Date': data['date'][i],
                            'Name': data['name'][i],
                            'Tweets': data['tweet'][i],
                            })


def get_all_dates(start_date, end_date):
    dates = []
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    step = timedelta(days=1)
    while start_date <= end_date:
        dates.append(str(start_date.date()))
        start_date += step

    return dates


def main():
    driver_type = int(input("1) Firefox | 2) Chrome | 3) IE | 4) Opera | 5) PhantomJS\nEnter the driver you want to use: "))
    wordsToSearch = input("Enter the words: ").split(',')
    for w in wordsToSearch:
        w = w.strip()
    start_date = input("Enter the start date in (YYYY-MM-DD): ")
    end_date = input("Enter the end date in (YYYY-MM-DD): ")

    lang = int(input("0) All Languages 1) English | 2) Italian | 3) Spanish | 4) French | 5) German | 6) Russian | 7) Chinese\nEnter the language you want to use: "))
    all_dates = get_all_dates(start_date, end_date)
    print(all_dates)
    write_csv_header()
    for i in range(len(all_dates) - 1):
        driver = init_driver(driver_type)
        scroll(driver, str(all_dates[i]), str(all_dates[i + 1]), wordsToSearch, lang)
        scrape_tweets(driver)
        time.sleep(5)
        print("The tweets for {} are ready!".format(all_dates[i]))
        driver.quit()


if __name__ == "__main__":
    main()
