# filename: search_gpt4_arxiv.py
import requests

def search_arxiv(query):
    url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': 1,
        'sortBy': 'lastUpdatedDate',
        'sortOrder': 'descending'
    }
    response = requests.get(url, params=params)
    return response.text

# Search for the latest paper about GPT-4
query = "GPT-4"
result = search_arxiv(query)
print(result)