import re
import requests

req = requests.get("https://github.com/Antar1011/Smogon-Usage-Stats/raw/59a9c1cf3570a9d68d89e073699cce17b1d999c7/TA.py")

groups = re.findall(r"tags\.append\(\'(.*)\'\)",req.content.decode())

print(list(set(groups)))