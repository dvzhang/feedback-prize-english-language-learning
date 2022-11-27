import os
from requests import get
import argparse
import requests
import random
from time import sleep
from lxml import etree

def UAPool():
    USER_AGENT_LIST = [
        		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.36 Safari/535.7',
        		'Mozilla/5.0 (Windows NT 6.2; Win64; x64; rv:16.0) Gecko/16.0 Firefox/16.0',
        		'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/534.55.3 (KHTML, like Gecko) Version/5.1.3 Safari/534.53.10'
        		"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        		"Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        		"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        		"Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        		"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        		"Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        		"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        		"Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        		"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        		"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.132 Safari/537.36",
        		"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:41.0) Gecko/20100101 Firefox/41.0"
        ]

    headers = {
                'User-Agent': USER_AGENT_LIST[int(random.random()*len(USER_AGENT_LIST))]
                }
    return headers


def scrapeOnePage(url):
    nextPage = ""
    if url[0] == "/":
        url = "https://lang-8.com/" + url
    res = requests.get(url, headers=UAPool())
    html = etree.HTML(res.text)

    ans = html.xpath('//li[@class="incorrect"]/text()')
    nextPages = html.xpath('//li[@class="pager_next"]/a/@href')
    if nextPages:
        nextPage = nextPages[0]
    print(ans)
    return ans, nextPage


def readUrl():
    with open("./data/url.txt", "r") as f:
        ans = f.readlines()
    
    for i in range(len(ans)):
        ans[i] = int(ans[i].split('/')[3])
    ans = list(set(ans))
    print(len(list(set(ans))))
    return list(set(ans))



def scrape(url):
    ans, nextPage = scrapeOnePage(url)
    if nextPage != "":
        ans += scrape(nextPage)
    sleep(random.random()*30)
    return ans


def save(articles, path = "./data/lang8.csv"):
    with open(path, 'a') as f:
        for article in articles:
            f.write("{}\n".format(article.replace("\n","").replace("\r","")))


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for Scraper.")
    # parser.add_argument("--task_type", default="",
    #                     type=str, help="which glue task is training")
    parser.add_argument("--scrape", required=False, action='store_true',
                        help="This will scrape the data but return only 5 entries of each dataset.")
    parser.add_argument("--static", type=str, default="", required=False,
                        help="This will return the static dataset scraped from the web and stored in database or CSV ﬁle")
    args = parser.parse_args()

    print('*'*60)
    print('args.scrape \t {}'.format(args.scrape))
    print('args.static \t {}'.format(args.static))
    print('*'*60)

    if args.scrape:
        ans,nextPage = scrapeOnePage(url = 'https://lang-8.com/1/notebook')
        for i in range(len(ans)):
            print("{}\t{}".format(i+1, ans[i]))
    
    elif args.static != "":
        with open(args.static, "r") as f:
            articles = f.readlines()
        for line in articles[:50]:
            print(line.strip())
        print('*'*60)
        a = input("Do you want to see all the data? The file is very large! (y/n)")
        if a == 'y':
            for line in articles[50:]:
                print(line.strip())

    else:
        TobeScrapedList = ['https://lang-8.com/{}/notebook'.format(i) for i in readUrl()]
        print(len(TobeScrapedList))
        TobeAdded = []
        for TobeScraped in TobeScrapedList:
            ans = scrape(url = TobeScraped)
            TobeAdded += ans
            if len(TobeAdded) > 30:
                save(TobeAdded)
                TobeAdded = []

    
if __name__ == "__main__":
    main()
