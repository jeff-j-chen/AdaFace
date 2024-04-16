from unidecode import unidecode
import mwclient
import wikipedia
import requests
import json
import urllib.request 
from tqdm import tqdm
import time

with open("names.txt", "r") as f:
    names = f.read()

page_titles = []
def process_letter(l):
    global page_titles
    user_agent = 'Baseballer/0.2 (jeffc3141@gmail.com)'
    site = mwclient.Site('en.wikipedia.org', clients_useragent=user_agent)
    page = site.pages[f"List of Major League Baseball players ({l})"]
    
    if page.exists:
        links = page.links()
        for link in links:
            s = link.page_title.split(" ")
            if not (len(s) > 1 and unidecode((' '+s[0]+' '+s[1]+' ').lower()) in names):
                continue
            page_titles.append(link.page_title)

WIKI_REQUEST = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

def save_wiki_img(search_term):
    result = wikipedia.search(search_term, results = 1)
    wikipedia.set_lang('en')
    wkpage = wikipedia.WikipediaPage(title = result[0])
    title = wkpage.title
    response  = requests.get(WIKI_REQUEST+title)
    json_data = json.loads(response.text)
    img_link = list(json_data['query']['pages'].values())[0]['original']['source']
    urllib.request.urlretrieve(img_link, f"faces/{unidecode(search_term).lower()}.png") 

letters = ["A", "B", "C", "D", "E", "F", "G", "Ha", "He–Hi", "Ho–Hz", "I", "J", "Ka–Ki", "Kj–Kz", "La–Lh", "Li–Lz", "Ma", "Mc–Me", "Mi–My", "N", "O", "Pa–Pg", "Ph–Pz", "Q", "R", "Sa–Se", "Sf–So", "Sp–Sz", "Ta–Th", "Ti–Tz", "U", "V", "Wa–Wh", "Wi–Wz", "Z"]
pbar = tqdm(range(len(letters)))
for i in pbar:
    pbar.set_description(f"Processing {letters[i]}")
    process_letter(letters[i])
    time.sleep(0.4)

fails = []
pbar = tqdm(page_titles)
for title in pbar:
    pbar.set_description(f"Processing {title}")
    time.sleep(0.4)
    try:
        save_wiki_img(title)
    except:
        print('ERROR')
        fails.append(title)

print(fails)
print(len(fails))