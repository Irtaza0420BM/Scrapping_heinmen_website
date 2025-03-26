import requests


url="https://www.heinemann-shop.com/en/global/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response= requests.get(url, headers=headers)
print(response.text)

if response.status_code==200:
    with open("output.html", "w", encoding="utf-8") as file:
       file.write(response.text)
    print("html content saved output.html")
else:
    print(f"field to fetch html.{response.status_code}")