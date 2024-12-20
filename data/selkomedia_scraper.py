import requests
from bs4 import BeautifulSoup, NavigableString, Comment,Tag
from html import escape
import re
import pandas as pd
import urllib.parse
import os
import shutil

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 250)

OUTPUT_FOLDER = 'html_selkomedia' # !!!WARNING !!! This folder and all contents is deleted and overwritten for each run!
FILENAME = 'selkomedia_texts.pickle'

# Input: list of URLs
#https://www.selkomedia.fi/paikalliset/7319543   # juttu on kooste 2-3 uutisesta, huonosti yhteensopivat
#https://www.selkomedia.fi/paikalliset/6537263   # pitkä listaus tilastoja, ei varsinainen uutinen

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
https://www.selkomedia.fi/paikalliset/8110152
https://www.selkomedia.fi/paikalliset/8102237
https://www.selkomedia.fi/paikalliset/8101580
https://www.selkomedia.fi/paikalliset/8094921
https://www.selkomedia.fi/paikalliset/8094850
https://www.selkomedia.fi/paikalliset/8082671
https://www.selkomedia.fi/paikalliset/8082551
https://www.selkomedia.fi/paikalliset/8078631
https://www.selkomedia.fi/paikalliset/8075751
https://www.selkomedia.fi/paikalliset/8069987
https://www.selkomedia.fi/paikalliset/8069965
https://www.selkomedia.fi/paikalliset/8054146
https://www.selkomedia.fi/paikalliset/8054081
https://www.selkomedia.fi/paikalliset/8050387
https://www.selkomedia.fi/paikalliset/8034743
https://www.selkomedia.fi/paikalliset/8034556
https://www.selkomedia.fi/paikalliset/8032400
https://www.selkomedia.fi/paikalliset/8027546
https://www.selkomedia.fi/paikalliset/8023579
https://www.selkomedia.fi/paikalliset/8016440
https://www.selkomedia.fi/paikalliset/8015874
https://www.selkomedia.fi/paikalliset/8008063
https://www.selkomedia.fi/paikalliset/7998149
https://www.selkomedia.fi/paikalliset/7989608
https://www.selkomedia.fi/paikalliset/7947227
https://www.selkomedia.fi/paikalliset/7947159
https://www.selkomedia.fi/paikalliset/7883161
https://www.selkomedia.fi/paikalliset/7874112
https://www.selkomedia.fi/paikalliset/7854893
https://www.selkomedia.fi/paikalliset/7854793
https://www.selkomedia.fi/paikalliset/7846583
https://www.selkomedia.fi/paikalliset/7813143
https://www.selkomedia.fi/paikalliset/7803663
https://www.selkomedia.fi/paikalliset/7786262
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
'7988739':'https://avecmedia.fi/bisnes/englanti-yleistyy-ravintola-alan-tyokielena-ja-nain-kielimuuri-ylitetaan-tiskin-takana-bartender-randall-muller-speaks-about-working-in-finland/',
'7975642':'https://www.helsinginuutiset.fi/paikalliset/7970353',
'7974660':'https://www.vantaansanomat.fi/paikalliset/7956975',
'7968458':'https://www.helsinginuutiset.fi/paikalliset/7964617',
'7956467':'https://www.helsinginuutiset.fi/paikalliset/7948093',
'7955701':'https://www.helsinginuutiset.fi/paikalliset/7944211',
'7955430':'https://www.helsinginuutiset.fi/paikalliset/7947717',
'7947770':'https://www.vantaansanomat.fi/paikalliset/7942962',
'8084624':'https://avecmedia.fi/ura/syrjintaa-vitsailua-vakivaltaa-tallaista-on-rasismi-ravintola-alalla',
'8064893':'https://www.helsinginuutiset.fi/paikalliset/8055258'
}

try:
    df=pd.read_pickle(FILENAME)
except:

    headers = {
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    }

    # Initialize a list to store data dictionaries
    data_list = []
    failed = set()

    for url in url_list:
        print(f'processing link {url}')
        id = url.split('/')[-1]
        data = {
            'TYPE_A_URL': url,
            'TYPE_A_HTML': None,
            'TYPE_B_URL': None,
            'TYPE_B_HTML': None,
            'ID': id,
        }
        try:
            # Download TYPE A HTML content
            response_a = requests.get(url)
            response_a.raise_for_status()  # Check for HTTP errors
            data['TYPE_A_HTML'] = response_a.text

            # Parse the HTML to find the specific hyperlink
            soup = BeautifulSoup(response_a.text, 'html.parser')

            if id in manual_match:
                link = manual_match[id]
            else:
                try:
                    link = soup.find('a', string=lambda text: text and "Voit lukea alkuperäisen" in text)
                    assert link is not None
                except:
                    link = soup.find('a', string=lambda text: text and "vantaansanomat.fi/paikalliset/" in text)
                assert link is not None,'Linkkiä ei löydy!'
                link = link.attrs['href']

            # Resolve the full URL in case of a relative link
            type_b_url = requests.compat.urljoin(url, link)
            data['TYPE_B_URL'] = type_b_url

            # Download TYPE B HTML content
            response_b = requests.get(type_b_url,headers=headers)
            response_b.raise_for_status()
            data['TYPE_B_HTML'] = response_b.text

        except Exception as err:
            print(f'An error occurred while processing {url}: {err}')
            failed.add(id)

        # Append the data dictionary to the list
        data_list.append(data)

    # Create pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    df.to_pickle('selkomedia_texts.pickle')

    print('failed ids:\n')
    print('\n'.join(list(failed)))

# Display the DataFrame
print(df)

import re

unwanted_texts = [
    'Helsingin Uutisten sovellus',
    "Tämä juttu on julkaistu ensin",
    "Vantaan Sanomat kertoo, mistä kaupungissa puhutaan",
    "Vantaan Sanomat kertoo,",
    'Juttu on julkaistu ensin',
    'Juttu on ensin julkaistu',
    'Juttu on julkaistu ensimmäisenä',
    'Vantaan Sanomat on mukana',
    'Juttua muokattu',
    'Tämä juttu on julkaistu ensimmäisenä Vantaan',
    'Juttu on julkaistu alunperin ',
    'Vantaan Sanomat kertoo,',
    'Jaa tämä artikkeli',
    'Helsingin Uutiset kertoo',
    'Lisää näistä aiheista',
    'Artikkeli on versioitu',
    'Jutusta on korjattu',
    'More articles from this publication',
]

def clean_html(html_code):
    if isinstance(html_code, BeautifulSoup):
        soup = html_code
    elif isinstance(html_code, str):
        soup = BeautifulSoup(html_code, 'html.parser')
    else:
        # If html_code is neither a string nor a BeautifulSoup object, return empty string
        return ''

        # Remove unwanted tags, excluding 'img' tags within allowed 'figure' elements
        # First, remove unwanted tags except 'img'
    for tag in soup(['svg', 'video', 'audio', 'source', 'iframe', 'embed', 'object', 'script']):
        tag.decompose()  # Remove the tag from the tree

        # Now, remove 'img' tags that are not within allowed 'figure' elements
    for img_tag in soup.find_all('img'):
        # Check if parent is a <figure> with class containing 'figure' and 'featured'
        parent_figure = img_tag.find_parent('figure', class_=lambda x: x and 'figure' in x and 'featured' in x)
        if parent_figure:
            # Do not remove this img tag
            continue
        else:
            # Remove the img tag
            img_tag.decompose()

        # Remove links to images and media, excluding those within allowed 'figure' elements
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].lower()
        # List of image and media extensions
        media_extensions = [
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
            '.mp4', '.avi', '.mov', '.mp3', '.wav', '.flv',
            '.wmv', '.m4v', '.webm', '.ogg', '.ogv'
        ]
        if any(href.endswith(ext) for ext in media_extensions):
            # Check if this link is within an allowed 'figure' element
            parent_figure = a_tag.find_parent('figure', class_=lambda x: x and (('figure' in x and 'featured' in x) or ("entry-content" in x)))
            if parent_figure:
                # Do not remove this link
                continue
            else:
                # Remove the link but keep the text
                a_tag.replace_with(a_tag.get_text())

        # Remove social media sharing segments
        # Remove elements with classes containing 'share' or 'social'
    for share_tag in soup.find_all(class_=lambda x: x and ('share' in x.lower() or 'social' in x.lower())):
        share_tag.decompose()

        # Remove elements with ids containing 'share' or 'social'
    for share_tag in soup.find_all(id=lambda x: x and ('share' in x.lower() or 'social' in x.lower())):
        share_tag.decompose()

        # Remove elements that link to social media sites
    social_domains = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'wa.me', 'whatsapp.com',
                      'pinterest.com', 't.me']
    for a_tag in soup.find_all('a', href=True):
        if any(domain in a_tag['href'] for domain in social_domains):
            parent = a_tag.parent
            # Remove parent element if it contains only this link
            if len(parent.find_all('a', href=True)) == 1:
                parent.decompose()
            else:
                a_tag.decompose()

    # Remove content enclosed with "_aside" type code, e.g., <aside class="diks-article__aside">
    for aside_tag in soup.find_all('aside', class_=lambda x: x and '_aside' in x):
        aside_tag.decompose()

    # Remove articles with class containing '--read-also'
    for article_tag in soup.find_all('article', class_=lambda x: x and '--read-also' in x):
        article_tag.decompose()

    def find_element_with_text(soup, text):
        for element in soup.descendants:
            if isinstance(element, NavigableString):
                if text in element:
                    return element.parent
            elif element.string and text in element.string:
                return element
        return None

    for unwanted_text in unwanted_texts:
        # Find paragraphs whose text contains the unwanted text
        target_element = find_element_with_text(soup, unwanted_text)

        if target_element:
            # If it's a NavigableString, get its parent
            if isinstance(target_element, NavigableString):
                target_element = target_element.parent

            # Remove the target element and all its following siblings
            current = target_element
            while current:
                next_sibling = current.next_sibling
                if isinstance(current, NavigableString):
                    current.extract()
                else:
                    current.decompose()
                current = next_sibling

        # Remove elements with class containing 'article-box'
    for box_tag in soup.find_all(class_=lambda x: x and 'article-box' in x):
        box_tag.decompose()

    return str(soup)

def extract_article_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    article_text = {'title':'','headline':'','text':'','HTML':None}
    # Find the article content

    try:

        main_content = soup.find_all(class_=re.compile(r'diks-article diks-article--border-right'))[0]
        assert main_content is not None

        article_text['HTML'] = str(main_content)

        article_content = main_content.find_all(class_=re.compile(r'_headline'))[0]
        article_text['title'] = article_content.text.strip()

        article_content = main_content.find_all(class_=re.compile(r'_lead'))[0]
        article_text['headline'] = article_content.text.strip()

        article_content = main_content.find_all(class_=re.compile(r'_storyline'))[0]
        article_text['text'] = article_content.text.strip()

    except:

        try:
            main_content = soup.find_all(class_=re.compile(r'site-main'))[0]
            assert main_content is not None

            article_text['HTML'] = str(main_content)

            article_content = main_content.find_all(class_=re.compile(r'entry-title'))[0]
            article_text['title'] = article_content.text.strip()

            article_content = main_content.find_all(class_=re.compile(r'ingressi'))[0]
            article_text['headline'] = article_content.text.strip()

            article_content = main_content.find_all('p')
            article_text['text'] = '\n'.join([element.get_text(strip=True) for element in article_content])
        except:
            try:
                main_content = soup.find_all(class_=re.compile(r'Story__root'))[0]
                assert main_content is not None

                article_text['HTML'] = str(main_content)

                article_content = main_content.find_all(class_=re.compile(r'ProductHeading'))[0]
                article_text['title'] = article_content.text.strip()

                article_content = soup.find_all('p',class_=re.compile('Paragraph ProductParagrap'))
                article_text['headline'] = article_content[0].text
                article_text['text'] = '\n'.join([element.get_text(strip=True) for element in article_content[1:] if ('class=' not in element.text)])
            except:
                try:
                    main_content = soup.find_all(class_=re.compile(r'diks-article diks-article--featured'))[0]
                    assert main_content is not None

                    article_text['HTML'] = str(main_content)

                    article_content = main_content.find_all(class_=re.compile(r'_headline'))[0]
                    article_text['title'] = article_content.text.strip()

                    article_content = main_content.find_all(class_=re.compile(r'_lead'))[0]
                    article_text['headline'] = article_content.text.strip()

                    article_content = main_content.find_all(class_=re.compile(r'_storyline'))[0]
                    article_text['text'] = article_content.text.strip()
                except:
                    return None
    for x in unwanted_texts:
        ind = article_text['text'].lower().find(x.lower())
        if ind>-1:
            article_text['text'] = article_text['text'][:ind]

    return article_text

df['TYPE_A_TEXT']=None
df['TYPE_B_TEXT']=None
df['TYPE_A_TEXT_SIMPLE']=''
df['TYPE_B_TEXT_SIMPLE']=''

bad_rows = []
for k,row in enumerate(df.iterrows()):
    print('processing text {ind}, URL_A: {urla}, URL_B: {urlb}'.format(ind = k+1,urla=row[1]['TYPE_A_URL'],urlb=row[1]['TYPE_B_URL']))

    if 'www.lansivayla.fi' in row[1]['TYPE_B_URL']:
        print('!!! Regular news from Länsiväylä, skipping!')
        bad_rows+=[row[0]]
        continue

    t1 = extract_article_text(row[1]['TYPE_A_HTML'])
    if t1 is None:
        print(f'...failed to parse text A: {row[1]["TYPE_A_URL"]}')
    t2 = extract_article_text(row[1]['TYPE_B_HTML'])
    if t2 is None:
        print(f'...failed to parse text B: {row[1]["TYPE_B_URL"]}')
    df.at[row[0],'TYPE_A_TEXT'] = t1
    df.at[row[0],'TYPE_B_TEXT'] = t2

    df.at[row[0],'TYPE_A_TEXT_SIMPLE'] = t1['title'] + '\n' + t1['headline'] + '\n' + t1['text']
    df.at[row[0],'TYPE_B_TEXT_SIMPLE'] = t2['title'] + '\n' + t2['headline'] + '\n' + t2['text']

    df.at[row[0],'TYPE_A_TEXT_CLEANED'] = clean_html(t1['HTML'])
    df.at[row[0],'TYPE_B_TEXT_CLEANED'] = clean_html(t2['HTML'])

old_n = len(df)
df = df.drop(index=bad_rows)
new_n = len(df)
if old_n>new_n:
    print(f'dropped {old_n-new_n} items')

import html

def filewriter(content,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def combine_raw_news_html(news_a,url_a,news_b,url_b,output_file):

    def format_news(news):
        s1 = re.sub(r'\n{2,}', '\n',html.escape(news['title'])).replace('\n', '<br><br>')
        s2 = re.sub(r'\n{2,}', '\n',html.escape(news['headline'])).replace('\n', '<br><br>')
        s3 = re.sub(r'\n{2,}', '\n',html.escape(news['text'])).replace('\n', '<br><br>')

        return (
            f"<h1>{s1}</h1>"
            f"<h2>{s2}</h2>"
            f"<p>{s3}</p>"
        )

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Combined News</title>
        <style>
            body {{
                display: flex;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            .news-container {{
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                height: 100vh;
            }}
            .separator {{
                width: 1px;
                background-color: #ccc;
            }}
            h1 {{
                font-size: 24px;
                color: #1a5f7a;
            }}
            h2 {{
                font-size: 20px;
                color: #2c3e50;
            }}
            p {{
                font-size: 16px;
            }}
            .news-url {{
                font-size: 14px;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="news-container">
            <div class="news-url">
                <a href="{url_a}" target="_blank">{url_a}</a>
            </div>
            {format_news(news_a)}
        </div>
        <div class="separator"></div>
        <div class="news-container">
            <div class="news-url">
                <a href="{url_b}" target="_blank">{url_b}</a>
            </div>
            {format_news(news_b)}
        </div>
    </body>
    </html>
    """

    filewriter(html_template, output_file)

    print(f"Combined HTML file created: {output_file}")


def make_comparison_html(clean_htmlA,url_a,clean_htmlB,url_b,output_file):
    #clean_htmlA = clean_html(html_codeA)
    #clean_htmlB = clean_html(html_codeB)

    # HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    .container {{
        display: flex;
    }}
    .left, .right {{
        flex: 1;
        padding: 10px;
        overflow: auto;
    }}
    .left {{
        border-right: 1px solid #000;
    }}
    .news-url {{
        font-size: 14px;
        margin-bottom: 10px;
    }}
    </style>
    </head>
    <body>
    <div class="container">  
        <div class="left">
            <div class="news-url">
                <a href="{url_a}" target="_blank">{url_a}</a>
            </div>  
            {contentA}
        </div>
        <div class="right">
            <div class="news-url">
                <a href="{url_b}" target="_blank">{url_b}</a>
            </div>          
            {contentB}
        </div>
    </div>
    </body>
    </html>
    '''

    # Fill the template
    html_output = html_template.format(contentA=clean_htmlA, contentB=clean_htmlB,url_a=url_a,url_b=url_b)

    # Write to file
    filewriter(html_output, output_file)

    return clean_htmlA,clean_htmlB

def text_cleaner_and_tagger2(input_html_str):
    soup = BeautifulSoup(input_html_str, 'html.parser')

    # Remove unwanted elements
    # Info about writer and date
    for div in soup.find_all('div', class_='diks-byline__text-wrapper'):
        div.decompose()
    # Figure captions
    for figcaption in soup.find_all('figcaption'):
        figcaption.decompose()
    # Special blocked quotes
    for blockquote in soup.find_all('blockquote', class_=lambda x: x and 'diks-blockquote' in x):
        blockquote.decompose()
    # Figures
    for figure in soup.find_all('figure'):
        figure.decompose()
    # Divs with class 'article-extra-content'
    for div in soup.find_all('div', class_='article-extra-content'):
        div.decompose()
    # Remove scripts and styles
    for element in soup(['script', 'style']):
        element.decompose()
    # Remove comments
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Now process the content
    # Try to find the article
    article = soup.find('article')
    tagged_text = ''

    def process_node(node):
        output = ''
        if isinstance(node, NavigableString):
            # It's a text node
            text = str(node)
            output += text
        elif node.name == 'h1':
            # Title
            text = node.get_text(separator=' ', strip=True)
            ind = text.find('|')
            if ind>-1 and ind<len(text)*0.50:
                text = text[(ind+1):]
            output += '<title>' + text + '</title>\n'
        elif node.name == 'h2':
            # Subtitle
            text = node.get_text(separator=' ', strip=True)
            output += '<subtitle>' + text + '</subtitle>\n'
        elif node.name == 'p':
            # Paragraph
            # Check if it's a lead
            if 'diks-article__lead' in node.get('class', []) or 'ingressi' in node.get('class', []):
                text = node.get_text(separator=' ', strip=True)
                output += '<lead>' + text + '</lead>\n'
            else:
                # Check if it's a quote
                text = node.get_text(separator=' ', strip=True)
                text_stripped = text.lstrip()
                # Define the quote markers
                quote_markers = ('–', '–\u2009', '–&thinsp;', '”', '"', '’', '‘', '«', '»', '„', '“', '”','”','−')
                if any(text_stripped.startswith(marker) for marker in quote_markers):
                    # It's a quote
                    # Remove leading and trailing quotation marks or dashes
                    # Define characters to strip
                    quote_chars = '–—\u2009\u202F\u200A\u200B\u2060'
                    quote_text = text_stripped.strip(quote_chars)
                    output += '<quote>' + quote_text + '</quote>\n'
                else:
                    # Regular paragraph
                    output += text + '\n'
        elif node.name == 'span':
            if 'ingressi' in node.get('class', []):
                # Lead
                text = node.get_text(separator=' ', strip=True)
                output += '<lead>' + text + '</lead>\n'
            else:
                # Process children
                for child in node.contents:
                    output += process_node(child)
        elif node.name == 'strong':
            # Bolded word
            text = node.get_text(separator=' ', strip=True)
            output += text
        elif node.name == 'em':
            # Emphasized text
            text = node.get_text(separator=' ', strip=True)
            output += text
        elif node.name == 'br':
            output += '\n'
        else:
            # Process children
            for child in node.contents:
                output += process_node(child)
        return output

    if article:
        tagged_text = process_node(article)
    else:
        # No article tag, process body
        body = soup.find('body')
        if body:
            tagged_text = process_node(body)
        else:
            # Process the whole soup
            tagged_text = process_node(soup)

    # Clean up multiple spaces and newlines
    tagged_text = re.sub(r'\n\s*\n', '\n', tagged_text)  # Remove multiple newlines
    tagged_text = re.sub(r'[ \t]+', ' ', tagged_text)  # Replace multiple spaces with single space
    tagged_text = tagged_text.strip()

    return tagged_text

def tagged_text_to_colored_html(tagged_text):

    # Map of tags to their corresponding RGB colors
    tag_color_map = {
        'title': 'rgb(0, 0, 255)',        # Blue
        'lead': 'rgb(0, 128, 0)',         # Green
        'subtitle': 'rgb(255, 165, 0)',   # Orange
        'quote': 'rgb(255, 0, 0)',        # Red
    }

    # Function to process elements recursively
    def process_element(element):
        if isinstance(element, NavigableString):
            # Escape HTML special characters in text nodes
            text = escape(str(element))
            return text
        elif isinstance(element, Tag):
            tag_name = element.name

            # Process child elements
            content = ''.join([process_element(child) for child in element.contents])

            if tag_name in tag_color_map:
                # Get the color for the tag
                color = tag_color_map[tag_name]

                # Split content into lines
                lines = content.split('\n')

                # Wrap each line with <p></p>
                wrapped_lines = [f'<p>{line.strip()}</p>' for line in lines if line.strip()]

                # Join wrapped lines
                wrapped_content = '\n'.join(wrapped_lines)

                # Wrap with <div style="color:...;"> to apply color across all lines
                return f'<div style="color: {color};">\n{wrapped_content}\n</div>'

            else:
                # For other tags or content, process recursively
                return process_element_children(element)

        else:
            return ''

    def process_element_children(element):
        content = ''
        for child in element.contents:
            result = process_element(child)
            if result.strip():
                # Split content into lines
                lines = result.split('\n')
                # Wrap each line with <p></p>
                wrapped_lines = [f'<p>{line.strip()}</p>' for line in lines if line.strip()]
                content += '\n'.join(wrapped_lines)
        return content

    # Parse the entire tagged_text
    soup = BeautifulSoup(tagged_text, 'html.parser')

    # Process the entire content
    html_text = process_element(soup)

    # Create the color coding explanation
    color_coding_lines = ['<p><strong>Color coding:</strong></p>']
    for tag_name in ['title', 'lead', 'subtitle', 'quote']:
        color = tag_color_map.get(tag_name, 'black')
        # Capitalize the tag name for display
        display_name = tag_name.capitalize()
        color_coding_line = f'<span style="color: {color};"> {display_name}</span>'
        color_coding_lines.append(color_coding_line)
    color_coding_html = '\n'.join(color_coding_lines)

    # Wrap the content in a basic HTML structure, including the color coding explanation
    html_output = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tagged text comparison</title>
    <style>
        /* Optional: Add some styling to paragraphs */
        p {{
            margin-bottom: 1em; /* Adjust the spacing between paragraphs */
            font-family: Arial, sans-serif; /* Optional: Set a default font */
        }}
    </style>
</head>
<body>
{color_coding_html}
<p>-------------------------------</p>
{html_text}
</body>
</html>'''

    return html_output
# Check if folder exists
if os.path.exists(OUTPUT_FOLDER):
    # Delete the folder
    shutil.rmtree(OUTPUT_FOLDER)

# Create the folder
os.makedirs(OUTPUT_FOLDER + os.sep + "formatted" , exist_ok=True)
os.makedirs(OUTPUT_FOLDER + os.sep + "raw" , exist_ok=True)
os.makedirs(OUTPUT_FOLDER + os.sep + "simple_html" , exist_ok=True)
os.makedirs(OUTPUT_FOLDER + os.sep + "tagged_texts" , exist_ok=True)

from urllib.parse import urlparse
def extract_domain(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract the netloc (domain and subdomain) from the parsed URL
    netloc = parsed_url.netloc

    # Split by '.' and extract the second part (domain)
    domain_parts = netloc.split('.')

    # Return the domain (the part between 'www.' and the first TLD)
    if len(domain_parts) > 2:
        return domain_parts[1]
    elif len(domain_parts) == 2:
        return domain_parts[0]
    else:
        return None

for k,row in enumerate(df.iterrows()):
    print(f'processing text {k+1}')
    id = row[1]['ID']
    if row[1]['TYPE_A_URL'] in url_list:
        output_file = OUTPUT_FOLDER + os.sep + f"formatted\\combined_news_original_{id}.html"
        make_comparison_html(str(row[1]['TYPE_A_TEXT_CLEANED']),row[1]['TYPE_A_URL'],str(row[1]['TYPE_B_TEXT_CLEANED']),row[1]['TYPE_B_URL'], output_file)
        output_file = OUTPUT_FOLDER + os.sep + f"raw\\combined_news_rawtext_{id}.html"
        combine_raw_news_html(row[1]['TYPE_A_TEXT'],row[1]['TYPE_A_URL'], row[1]['TYPE_B_TEXT'],row[1]['TYPE_B_URL'], output_file)

        tagged_textA = text_cleaner_and_tagger2(str(row[1]['TYPE_A_TEXT_CLEANED']))
        output_file = OUTPUT_FOLDER + os.sep + f"tagged_texts\{extract_domain(row[1]['TYPE_A_URL'])}_selko_{id}.txt"
        filewriter(tagged_textA,output_file)
        output_file = OUTPUT_FOLDER + os.sep + f"tagged_texts\\{extract_domain(row[1]['TYPE_A_URL'])}_selko_comparison_{id}.html"
        make_comparison_html(str(row[1]['TYPE_A_TEXT_CLEANED']).replace('.jpg','.dummy').replace('.png','.dummy'),row[1]['TYPE_A_URL'],tagged_text_to_colored_html(tagged_textA),'tagged text', output_file)

        tagged_textB = text_cleaner_and_tagger2(str(row[1]['TYPE_B_TEXT_CLEANED']))
        output_file = OUTPUT_FOLDER + os.sep + f"tagged_texts\\{extract_domain(row[1]['TYPE_B_URL'])}_regular_{id}.txt"
        filewriter(tagged_textB,output_file)
        output_file = OUTPUT_FOLDER + os.sep + f"tagged_texts\\{extract_domain(row[1]['TYPE_B_URL'])}_regular_comparison_{id}.html"
        make_comparison_html(str(row[1]['TYPE_B_TEXT_CLEANED']).replace('.jpg','.dummy').replace('.png','.dummy'),row[1]['TYPE_B_URL'],tagged_text_to_colored_html(tagged_textB),'tagged text', output_file)

        output_file = OUTPUT_FOLDER + os.sep + f"simple_html\\selko_{id}.html"
        filewriter(str(row[1]['TYPE_A_TEXT_CLEANED']),output_file)

        output_file = OUTPUT_FOLDER + os.sep + f"simple_html\\regular_{id}.html"
        filewriter(str(row[1]['TYPE_B_TEXT_CLEANED']),output_file)

df.to_pickle(FILENAME)

print('\nall done!')
