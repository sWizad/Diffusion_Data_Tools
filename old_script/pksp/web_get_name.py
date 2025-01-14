import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict
from fastprogress import progress_bar

def scrape_fusion_dex() -> Dict[str, str]:
    """
    Scrapes the Fusion Dex website for Pokemon fusion names.
    Returns a dictionary with URL numbers as keys and fusion names as values.
    """
    base_url = "https://www.fusiondex.org/{}"
    results = {}
    
    # Adding headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Loop through pages 1 to 470
    for page_num in progress_bar(range(1, 471)):
        try:
            # Construct the URL
            url = base_url.format(page_num)
            
            # Add a delay to be respectful to the server
            time.sleep(1)
            
            # Make the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the specific element
            dex_entry = soup.find(class_='dex-entry sprite-variant-main')
            if dex_entry:
                header = dex_entry.find('header')
                if header:
                    h2 = header.find('h2')
                    if h2:
                        # Store the result
                        results[str(page_num)] = h2.text.strip()
                        print(f"Successfully scraped page {page_num}: {h2.text.strip()}")
                    else:
                        print(f"No h2 found on page {page_num}")
                else:
                    print(f"No header found on page {page_num}")
            else:
                print(f"No dex-entry found on page {page_num}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error scraping page {page_num}: {e}")
            continue
            
    return results

def save_to_json(data: Dict[str, str], filename: str = "fusion_dex_results.json") -> None:
    """
    Saves the scraped data to a JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    print("Starting the scraping process...")
    results = scrape_fusion_dex()
    
    print(f"\nScraping completed. Found {len(results)} entries.")
    
    # Save results to JSON
    save_to_json(results)
    print(f"\nResults have been saved to fusion_dex_results.json")