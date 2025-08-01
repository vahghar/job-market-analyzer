import requests
import pandas as pd

# Your API credentials
APP_ID = "a4370991"
APP_KEY = "bccde9af2756d92cfffaef55203dbf9c"

BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search/"

def fetch_jobs(role, location, pages=3):
    all_jobs = []

    for page in range(1, pages + 1):
        url = f"{BASE_URL}{page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "what": role,
            "where": location,
            "results_per_page": 20,
            "content-type": "application/json"
        }

        response = requests.get(url, params=params)
        print(f"Fetching page {page} | Status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Failed to fetch page {page}")
            continue

        data = response.json()
        jobs = data.get("results", [])
        for job in jobs:
            all_jobs.append({
                "role": role,
                "location": location,
                "title": job.get("title"),
                "company": job.get("company", {}).get("display_name"),
                "description": job.get("description")[:300], 
                "created": job.get("created")
            })

    return all_jobs

if __name__ == "__main__":
    roles = ["data analyst", "business analyst", "software engineer"]
    cities = ["Delhi", "Bangalore", "Mumbai"]

    all_results = []

    for role in roles:
        for city in cities:
            print(f"\n🔍 Scraping: {role.title()} in {city}")
            jobs = fetch_jobs(role, city, pages=3)
            all_results.extend(jobs)

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("adzuna_all_jobs.csv", index=False)
    print(f"\n✅ Done! Scraped {len(df)} total jobs.")
