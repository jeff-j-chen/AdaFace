#This will not run on online IDE
import requests
from bs4 import BeautifulSoup

# URL = "https://www.google.com/search?q=david+aardsma";
# r = requests.get(URL)

# soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
# print(soup.prettify())
response = requests.get("http://free-proxy-list.net")
soup = BeautifulSoup(response.content, "lxml")
table = soup.find("table", id="proxylisttable")
for tr in table.tbody.find_all("tr"):
    info = tr.find_all("td")
    if info[4].string != "elite proxy":
        continue
    if info[6].string == "yes":
        protocol = "https"
    else:
        protocol = "http"
    addr = f"{info[0].string}:{info[1].string}"
    print({"addr": addr, "protocol": protocol})