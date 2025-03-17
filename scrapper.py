import requests
from bs4 import BeautifulSoup
import os
import re
import time

# Wikipedia URLs for Special Forces
SPECIAL_FORCES = {
    "Garud Commando Force": "https://en.wikipedia.org/wiki/Garud_Commando_Force",
    "Para SF": "https://en.wikipedia.org/wiki/Para_(Special_Forces)",
    "MARCOS": "https://en.wikipedia.org/wiki/MARCOS",
    "National Security Guard": "https://en.wikipedia.org/wiki/National_Security_Guard",
    "special protection group": "https://en.wikipedia.org/wiki/Special_Protection_Group",
    "special frontier force": "https://en.wikipedia.org/wiki/Special_Frontier_Force",
    "51 Special Action Group": "https://en.wikipedia.org/wiki/51_Special_Action_Group",
    "Special Group (India)": "https://en.wikipedia.org/wiki/Special_Group_(India)",
}

# Headers to mimic a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Create a folder to store text files
os.makedirs("special_forces_text", exist_ok=True)

# Function to clean extracted text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
    text = re.sub(r'\[.*?\]', '', text)  # Remove citation references like [1]
    text = text.replace("\n", " ").strip()
    return text

# Function to scrape and save Wikipedia pages
def scrape_special_forces():
    for index, (force_name, url) in enumerate(SPECIAL_FORCES.items(), start=1):
        retries = 3  # Allow 3 retry attempts
        while retries > 0:
            try:
                print(f"Fetching: {url} (Attempt {4 - retries}/3)")
                response = requests.get(url, headers=HEADERS, timeout=10)
                if response.status_code == 200:
                    break
            except requests.exceptions.Timeout:
                print(f"Timeout for {force_name} ({url})")
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            
            retries -= 1
            time.sleep(5)
        
        if retries == 0:
            print(f"Skipping {force_name} due to repeated failures.")
            continue
        
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        full_text = clean_text(content_div.get_text(separator=" ")) if content_div else "No content found"
        
        filename = f"special_forces_text/{index}_{force_name.replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Special Force: {force_name}\n")
            f.write(f"Wiki URL: {url}\n\n")
            f.write(full_text)
        
        print(f"Saved: {filename}")

    print("Scraping completed!")

# Run the scraper
scrape_special_forces()
print("Scraping completed! Check 'special_forces_text' folder.")
