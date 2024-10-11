from tqdm import tqdm
import time
import random
from custom_crawler import MyImageCrawler, MyImageDownloader
user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.2420.81", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"]

with open("names.txt", "r") as f:
    names = f.readlines()

my_crawler = MyImageCrawler(
    downloader_cls=MyImageDownloader,
    storage = {'root_dir': r'faces_google'}
)
name = ""
prog = tqdm(range(len(names)))
for i in prog:
    name = names[i].strip()
    if i % 50 == 0:
        MyImageCrawler.set_session(my_crawler, {"User-Agent": (random.choice(user_agents))})
    prog.set_description(f"tqdm: {name}")
    time.sleep(0.5)
    query = f'{name.strip()} baseball player'
    my_crawler.crawl(keyword=query, person_name=name.strip().replace(" ", "_"), max_num=3)
    time.sleep(0.5)