from icrawler.builtin import BingImageCrawler, BaiduImageCrawler, GoogleImageCrawler
from icrawler import ImageDownloader
from icrawler import Parser
from tqdm import tqdm
from urllib.parse import urlparse
import uuid
import logging
import time
from bs4 import BeautifulSoup
import re
from icrawler.utils import ProxyPool, Proxy
import sys
import random
user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.2420.81", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36", "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"]

with open("names.txt", "r") as f:
    names = f.readlines()


class MyImageDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        global file_names
        url_path = urlparse(task['file_url'])[2]
        extension = url_path.split('.')[-1] if '.' in url_path else default_ext
        with open('current_player.txt', "r") as f:
            cur_name = '_'.join(f.read().strip().split(' '))
            print(f"read: {cur_name}")
        file_idx = self.fetched_num + self.file_idx_offset
        print(f"fetched_num: {self.fetched_num}, file_idx: {self.file_idx_offset}")
        if self.fetched_num > 3:
            return "PLACEHOLDER.png"
        return f"{cur_name}_{file_idx}.{extension}"
    
class MyParser(Parser):
    def parse(self, response):
        soup = BeautifulSoup(response.content.decode("utf-8", "ignore"), "lxml")
        image_divs = soup.find_all(name="script")
        for div in image_divs:
            txt = str(div)

            uris = re.findall(r"http[^\[]*?.(?:jpg|png|bmp)", txt)
            if not uris:
                uris = re.findall(r"http[^\[]*?\.(?:jpg|png|bmp)", txt)
            uris = [bytes(uri, "utf-8").decode("unicode-escape") for uri in uris]
            if uris is not None:
                return [{"file_url": uri} for uri in uris]
            else:
                print("got none back")
                sys.exit()
    

class MyCrawler(GoogleImageCrawler):

    def set_proxy_pool(self, pool=None):
        self.proxy_pool = ProxyPool()
        self.proxy_pool.add_proxy(Proxy("brd-customer-hl_3d243eb6-zone-residential_proxy1:mo3kz3k66e8s@brd.superproxy.io:22225", "http"))

my_crawler = MyCrawler(
    downloader_cls=MyImageDownloader,
    # parser_cls=MyParser,
    storage = {'root_dir': r'faces_google'}
)

prog = tqdm(range(14565, len(names)))
for i in prog:
    name = names[i].strip()
    if i % 50 == 0:
        MyCrawler.set_session(my_crawler, {"User-Agent": (random.choice(user_agents))})
    prog.set_description(f"tqdm: {name}")
    time.sleep(0.25)
    with open('current_player.txt', "w") as f:
        f.write(name)
    print(f"\nwrote: {name}")
    query = f'{name.strip()} baseball player'
    my_crawler.crawl(keyword=query, max_num=2)
    time.sleep(0.25)