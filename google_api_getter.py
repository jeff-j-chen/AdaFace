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
            if uris:
                return [{"file_url": uri} for uri in uris]
    

bing_crawler = GoogleImageCrawler(
    downloader_cls=MyImageDownloader,
    parser_cls=MyParser
    storage = {'root_dir': r'faces_google'}
)

prog = tqdm(range(45, len(names)))
# prog = tqdm([i for i in range(10)])
for i in prog:
    # if i > 0: break
    name = names[i].strip()
    prog.set_description(f"tqdm: {name}")
    time.sleep(0.25)
    with open('current_player.txt', "w") as f:
        f.write(name)
    print(f"\nwrote: {name}")
    query = f'{name.strip()} baseball player'
    bing_crawler.crawl(keyword=query, max_num=3)
    time.sleep(0.25)