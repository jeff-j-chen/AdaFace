from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': 'your_image_dir'})
google_crawler.crawl(keyword='cat', offset=0, max_num=10,
                     min_size=(200,200), max_size=None, file_idx_offset=0)