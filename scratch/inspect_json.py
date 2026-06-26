import requests
import json

url = "https://www.transfermarkt.es/ticker/begegnung/live/4776650/json"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

resp = requests.get(url, headers=headers)
data = resp.json()

# Save data to scratch/json_sample.json for inspection
import os
os.makedirs("scratch", exist_ok=True)
with open("scratch/json_sample.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("JSON Keys:", list(data.keys()))
if 'aufstellung' in data:
    print("aufstellung keys:", list(data['aufstellung'].keys()))
    # print some keys
    for k in ['heim', 'gast']:
        if k in data['aufstellung']:
            print(f"  {k} keys:", list(data['aufstellung'][k].keys()))
            # check starts
            if 'start' in data['aufstellung'][k]:
                print(f"    {k} start players count:", len(data['aufstellung'][k]['start']))
                # print first player details
                if data['aufstellung'][k]['start']:
                    print(f"      First player:", data['aufstellung'][k]['start'][0])
else:
    # Print keys recursively or search for dzeko
    def search_dict(d, q, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if q.lower() in str(k).lower():
                    print(f"Key match: {path}.{k}")
                search_dict(v, q, f"{path}.{k}")
        elif isinstance(d, list):
            for i, x in enumerate(d):
                search_dict(x, q, f"{path}[{i}]")
        elif isinstance(d, str):
            if q.lower() in d.lower():
                print(f"Value match in {path}: {d}")

    search_dict(data, "dzeko")
