import requests
from bs4 import BeautifulSoup
import json

def scrape_fusiondex():
    # URL of the website
    url = "https://www.fusiondex.org/"
    
    # Dictionary to store results
    link_dict = {}
    
    try:
        # Send GET request to the website
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all <a> tags with tabindex="-1"
        links = soup.find_all('a', attrs={'tagindex': '-1'})
        
        # Process each link
        for link in links:
            href = link.get('href', '')
            # Remove leading and trailing slashes to get the key
            key = href.strip('/')
            if key:  # Only add if we have a valid key
                # Get the text content of the link
                link_text = link.get_text(strip=True)
                link_dict[key] = link_text
        
        # Save to JSON file
        with open('fusiondex_links.json', 'w', encoding='utf-8') as f:
            json.dump(link_dict, f, indent=4, ensure_ascii=False)
            
        print("Data successfully saved to fusiondex_links.json")
        return link_dict
        
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    result = scrape_fusiondex()
    if result:
        print("\nSample of collected data:")
        # Print first few items as example
        for key, value in list(result.items())[:3]:
            print(f"Key: {key}, Text: {value}")