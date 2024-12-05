import requests
from bs4 import BeautifulSoup, NavigableString, Comment,Tag
from html import escape
import re
import pandas as pd
import urllib.parse
import os
import shutil
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 250)

url_list = '''
https://www.selkomedia.fi/paikalliset/7975642
https://www.selkomedia.fi/paikalliset/7846765
https://www.selkomedia.fi/paikalliset/7839230
https://www.selkomedia.fi/paikalliset/7837855
https://www.selkomedia.fi/paikalliset/7828874
https://www.selkomedia.fi/paikalliset/7820792
https://www.selkomedia.fi/paikalliset/7813418
https://www.selkomedia.fi/paikalliset/7344172
https://www.selkomedia.fi/paikalliset/7325109
https://www.selkomedia.fi/paikalliset/7324246
https://www.selkomedia.fi/paikalliset/7318389
https://www.selkomedia.fi/paikalliset/7310949
https://www.selkomedia.fi/paikalliset/7296887
https://www.selkomedia.fi/paikalliset/6821082
https://www.selkomedia.fi/paikalliset/7803134
https://www.selkomedia.fi/paikalliset/6827761
https://www.selkomedia.fi/paikalliset/6702850
https://www.selkomedia.fi/paikalliset/6687371
https://www.selkomedia.fi/paikalliset/6656178
https://www.selkomedia.fi/paikalliset/6626222
https://www.selkomedia.fi/paikalliset/6621560
https://www.selkomedia.fi/paikalliset/6595747
https://www.selkomedia.fi/paikalliset/7800155
https://www.selkomedia.fi/paikalliset/7728724
https://www.selkomedia.fi/paikalliset/7343889
https://www.selkomedia.fi/paikalliset/6866533
https://www.selkomedia.fi/paikalliset/6702568
https://www.selkomedia.fi/paikalliset/6651721
https://www.selkomedia.fi/paikalliset/6635280
https://www.selkomedia.fi/paikalliset/6622634
https://www.selkomedia.fi/paikalliset/7354783
https://www.selkomedia.fi/paikalliset/7333624
https://www.selkomedia.fi/paikalliset/7289866
https://www.selkomedia.fi/paikalliset/6851848
https://www.selkomedia.fi/paikalliset/6839120
https://www.selkomedia.fi/paikalliset/6809100
https://www.selkomedia.fi/paikalliset/6787804
https://www.selkomedia.fi/paikalliset/6780541
https://www.selkomedia.fi/paikalliset/6754513
https://www.selkomedia.fi/paikalliset/7829167
https://www.selkomedia.fi/paikalliset/7820396
https://www.selkomedia.fi/paikalliset/7347916
https://www.selkomedia.fi/paikalliset/7343584
https://www.selkomedia.fi/paikalliset/7988739
https://www.selkomedia.fi/paikalliset/7974660
https://www.selkomedia.fi/paikalliset/7968458
https://www.selkomedia.fi/paikalliset/7956467
https://www.selkomedia.fi/paikalliset/7955701
https://www.selkomedia.fi/paikalliset/7955430
https://www.selkomedia.fi/paikalliset/7947770
'''
url_list = [x for x in url_list.split('\n') if len(x)>0]

# for those articles where automatic linking is not available
manual_match = {
'6809100':'https://www.helsinginuutiset.fi/paikalliset/6805395',
'6754513':'https://www.vantaansanomat.fi/paikalliset/6753053',
'6821082':'https://avecmedia.fi/bisnes/suoraa-ja-epasuoraa-rasismia-kohdannut-manageri-lilli-keh-antaa-aanen-ruskeille-naisille/',
'6780541':'https://www.vantaansanomat.fi/paikalliset/6773779',
'6537263':'https://www.vantaansanomat.fi/paikalliset/6528761',
'6595747':'https://www.vantaansanomat.fi/paikalliset/6584015',
'6622634':'https://www.helsinginuutiset.fi/paikalliset/6894862',
'6651721':'https://www.vantaansanomat.fi/paikalliset/6644169',
'6702568':'https://issuu.com/myyntijamarkkinointi/docs/223mma/s/24556977',
#'7319543':'https://www.vantaansanomat.fi/paikalliset/7315097' # ainakin 3 tekstiä samassa
'7988739':'https://avecmedia.fi/bisnes/englanti-yleistyy-ravintola-alan-tyokielena-ja-nain-kielimuuri-ylitetaan-tiskin-takana-bartender-randall-muller-speaks-about-working-in-finland/',
'7975642':'https://www.helsinginuutiset.fi/paikalliset/7970353',
'7974660':'https://www.vantaansanomat.fi/paikalliset/7956975',
'7968458':'https://www.helsinginuutiset.fi/paikalliset/7964617',
'7956467':'https://www.helsinginuutiset.fi/paikalliset/7948093',
'7955701':'https://www.helsinginuutiset.fi/paikalliset/7944211',
'7955430':'https://www.helsinginuutiset.fi/paikalliset/7947717',
'7947770':'https://www.vantaansanomat.fi/paikalliset/7942962'
}
url_list = url_list + [r'https://www.selkomedia.fi/paikalliset/'+k for k in manual_match.keys()]

article_links = []
for page in range(1,5):

    print(f'Page {page}')
    url = f'https://www.selkomedia.fi/uusimmat/sivu/{page}'
    response_a = requests.get(url)
    response_a.raise_for_status()  # Check for HTTP errors

    # Parse the HTML content
    soup = BeautifulSoup(response_a.text, 'html.parser')

    # Find all <a> tags with the specified class
    links = soup.find_all('a', class_='diks-card__link')

    # Extract and print the href attribute for each <a> tag
    for link in links:
        href = link.get('href')
        print(href)
        article_links.append(href)

article_links = [x for x in article_links if x not in url_list]

print('\n\nAnalyzing links\n')

paired_samples = []
manual_match = []
for url in article_links:

    response_a = requests.get(url)
    response_a.raise_for_status()  # Check for HTTP errors

    # Parse the HTML content
    soup = BeautifulSoup(response_a.text, 'html.parser')

    published_time = soup.find('time', class_='diks-date__published')

    published_datetime='NOTFOUND'
    if published_time:
        published_datetime = published_time.get('datetime')

    html = str(response_a.text).lower()

    if 'juttu on julkaistu ensimmäisenä' in html or 'voit lukea alkuperäisen' in html or 'muutettu selkokielelle' in html:
        print(f'...found selko for news published at {published_datetime}: {url}')

        links = soup.find_all('a', string=lambda text: text and "lukea alkuperäisen jutun" in text)
        if len(links) != 1:
            print(f'incorrect link count ({len(links)}) !! Need manual matching')
            manual_match+=[url]
            continue

        for link in links:
            href = link.get('href')
            paired_samples.append({'regular':href,'selko':url})

print(f'\nExtracted total {len(paired_samples)} paired samples\n')

print('valid links:\n')
for sample in paired_samples:
    print(f"{sample['selko']}")

print('\nfind manually for these:\n')
for x in manual_match:
    print(f"{x}")

print('\n\n all done!')


#'<p><em>Juttu on julkaistu ensimmäisenä Helsingin Uutisissa. Se on muutettu selkokielelle tekoälyn avulla.</em> <a href="https://www.helsinginuutiset.fi/paikalliset/8098022">Voit lukea alkuperäisen jutun täältä.</a></p>'