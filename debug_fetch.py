import requests

def debug_session_fetch(draw_no):
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://dhlottery.co.kr/',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    session.headers.update(headers)
    
    try:
        # Step 1: Visit homepage to get cookies/session
        print("Visiting homepage...")
        resp_home = session.get('https://dhlottery.co.kr/common.do?method=main')
        print(f"Homepage Status: {resp_home.status_code}")
        
        # Step 2: Try JSON API
        print("Trying JSON API...")
        api_url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
        resp_api = session.get(api_url)
        print(f"API Status: {resp_api.status_code}")
        try:
            print("API Response JSON:", resp_api.json())
        except:
            print("API Response is not JSON")
            print(resp_api.text[:500])
            
        # Step 3: Try Scraping Result Page
        print("Trying Result Page...")
        scrape_url = f"https://dhlottery.co.kr/gameResult.do?method=byWin&drwNo={draw_no}"
        resp_scrape = session.get(scrape_url)
        print(f"Scrape Status: {resp_scrape.status_code}")
        if 'ball_645' in resp_scrape.text:
            print("Success! Found ball_645 in scraping result.")
            # Print content around first ball
            idx = resp_scrape.text.find('ball_645')
            print(resp_scrape.text[idx:idx+200])
        else:
            print("Failed to find numbers in scrape result.")
            print(resp_scrape.text[:500])

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_session_fetch(1205)
