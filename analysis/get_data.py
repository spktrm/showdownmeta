import json
import requests

FILES = {
    "abilities",
    "conditions",
    "items",
    "moves",
    "natures",
    "species",
}


def main():
    for file in FILES:
        data = requests.get(f"https://github.com/pkmn/ps/raw/main/dex/data/{file}.json")
        with open(f"analysis/{file}.json", "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
