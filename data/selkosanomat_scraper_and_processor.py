import time
import requests
from bs4 import BeautifulSoup,NavigableString
import pandas as pd
import urllib.parse
import os
import shutil
import re
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 250)

FILENAME_html = 'selkosanomat_news_htmls.pickle'
FILENAME_url = 'selkosanomat_news_urls.pickle'

def extract_articles(html_code):
    # Create BeautifulSoup object
    soup = BeautifulSoup(html_code, 'html.parser')

    # Find all div elements with class "inside-article"
    articles = soup.find_all('div', class_='inside-article')

    # Extract URLs from entry-summary divs within inside-article
    article_urls = []
    for article in articles:
        summary = article.find('div', class_='entry-summary')
        if summary:
            link = summary.find('a')
            if link and link.get('href'):
                article_urls.append(link['href'])

    return article_urls

print(f'fetching URLs')
try:
    all_urls = pd.read_pickle(FILENAME_url)
except:
    categories = ['suomi','maailma','kulttuuri','arki']
    years = range(2014,2025)
    all_urls = pd.DataFrame([])
    for year in years:
        for category in categories:
            print(f'...processing {category} {year}',end='')
            page=1
            while 1:
                url = f'https://selkosanomat.fi/arkisto/?sivu={page}&kategoria={category}&vuosi={year}'
                response = requests.get(url)
                response.raise_for_status()  # Check for HTTP errors
                article_urls = extract_articles(response.text)
                if len(article_urls)==0 or page>220:
                    break
                else:
                    for u in article_urls:
                        row = {'category':category,'year':year,'url':u}
                        all_urls=pd.concat([all_urls,pd.DataFrame(row,index=[0])],ignore_index=True)
                page+=1
            print(f'.. saved {page} pages')
    all_urls.to_pickle(FILENAME_url)

try:
    DATA = pd.read_pickle(FILENAME_html)
except:
    DATA = pd.DataFrame([],columns=['category','year','url','html'])

processed = set(DATA['url'].to_list())
print(f'fetching articles ({len(processed)} already included)')
passed,added = 0,0
N = len(all_urls)
for row in all_urls.iterrows():
    url = row[1]['url']
    if url in processed:
        passed+=1
        print(f'...passed {url} ({passed+added} or {N})')
        pass
    else:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        html = response.text
        r = row[1].to_dict()
        r['html'] = html
        DATA = pd.concat([DATA,pd.DataFrame(r,index=[0])],ignore_index=True)
        DATA.to_pickle(FILENAME_html)
        added+=1
        time.sleep(0.1 + np.random.rand(1)[0])
        print(f'...ADDED {url} ({passed+added} or {N})')

def process_html(html_str):
    try:
        # Create BeautifulSoup object
        soup = BeautifulSoup(html_str, 'html.parser')

        # Initialize output text and tracking variables
        output = []
        title_count = 0
        lead_count = 0
        subtitle_count = 0
        total_text_length = 0
        last_subtitle_position = -1
        subtitle_char_count = 0
        regular_text_char_count = 0
        quote_count=0

        # Check for unexpected heading structure
        all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading_levels = set(h.name for h in all_headings)

        if 'h1' not in heading_levels:
            print("Warning: No h1 heading found")
            return -1

        if len(heading_levels) > 3:  # Too many different heading levels
            print("Warning: Too many heading levels used")
            return -1

        # Find article content
        article = soup.find('article')
        if not article:
            print("Warning: No article content found")
            return -1

        # Extract title
        title = article.find('h1', class_='entry-title')
        if title:
            title_text = title.get_text().strip()
            output.append(f"<title>{title_text}</title>")
            title_count += 1
            total_text_length += len(title_text)

        # Extract lead/ingress
        lead = article.find('div', class_='single-excerpt')
        if lead:
            lead_text = lead.get_text().strip()
            output.append(f"<lead>{lead_text}</lead>")
            lead_count += 1
            total_text_length += len(lead_text)

        # Find main content
        content = article.find('div', class_='entry-content')
        if not content:
            print("Warning: No main content found")
            return -1

        # Process paragraphs
        for idx, elem in enumerate(content.find_all(['p', 'h2', 'h3', 'h4'])):
            text = elem.get_text().strip()
            if not text:
                continue

            # Skip certain elements
            if elem.parent.get('class') and any(cls in ['rs_skip', 'info-row', 'article-end-buttons']
                                                for cls in elem.parent.get('class')):
                continue

            total_text_length += len(text)

            # Process subtitles
            is_subtitle = False

            if elem.name in ['h2','h3','h4']:
                is_subtitle = True

            # Check for strong-only paragraphs
            elif elem.name == 'p' and elem.find('strong'):
                strong_text = elem.find('strong').get_text().strip()
                if (strong_text == text and
                        len(text) < 150 and
                        len(text.split('.')) <= 2):
                    is_subtitle = True

            if is_subtitle:
                if last_subtitle_position != -1 and idx - last_subtitle_position < 2:
                    output.append(text)
                    regular_text_char_count += len(text)
                else:
                    output.append(f"<subtitle>{text}</subtitle>")
                    subtitle_char_count += len(text)
                    subtitle_count += 1
                    last_subtitle_position = idx
                continue

            text_cmp = text.strip()
            if (re.match(r'^[\s]*[\u2011\u2012\u2013\u2014\u2015\u2212\u2E3A\u2E3B\u2043\u002D\u2010-]+', text_cmp) or  # Various dash characters
                (text_cmp.startswith('"') and text_cmp.endswith('"')) and text_cmp.count('"')==2):
                output.append(f"<quote>{text}</quote>")
                quote_count += 1
            else:
                output.append(text)

            regular_text_char_count += len(text)

        # Perform sanity checks
        if title_count == 0:
            print("Warning: No title found in article")
            return -1
        if title_count > 1:
            print("Warning: Multiple titles found in article")
            return -1
        if lead_count > 1:
            print("Warning: Multiple leads found in article")
            return -1

        subtitle_ratio = subtitle_char_count / total_text_length

        # Check for excessive whitespace or very short paragraphs
        filtered_output = []
        for line in output:
            line = line.strip()
            if len(line) > 0:  # Minimum length for a meaningful paragraph
                filtered_output.append(line)

        # Final sanity check on filtered output
        if len(filtered_output) < 3:
            print("Warning: Article contains too few meaningful paragraphs")
            return -1

        # Check if the article seems too short after filtering
        total_filtered_chars = sum(len(line) for line in filtered_output)
        total_original_chars = sum(len(line) for line in output)
        if total_filtered_chars < total_original_chars*0.95:  # Minimum reasonable article length
            print("Warning: Article text is too short after filtering")
            return -1

        return {'tagged_text':"\n".join(filtered_output),
                'subtitle_ratio':subtitle_ratio,
                'total_text_length':total_text_length,
                'subtitle_count':subtitle_count,
                'quote_count':quote_count,
                'article_html':str(article),
                'content_html':str(content)}

    except Exception as e:
        print(f"Error processing article: {str(e)}")
        return '',-1


print(f'processing articles')
DATA['html_length'] = DATA['html'].apply(lambda x:len(x))
DATA = DATA.sort_values(by='html_length',ascending=False)
DATA = DATA.sample(frac=1.0)

urls=[
    'https://selkosanomat.fi/maailma/apu-kulkee-hitaasti-nepaliin/',
]
#DATA = DATA[DATA['url'].isin(urls)]

print(f'full data size {len(DATA)}')
flag=False
for row in DATA.iterrows():
    html = row[1]['html']
    res = process_html(html)
    if res == -1:
        print(f"...failed to process {row[1]['url']}")
    else:
        if not flag:
            for key in res.keys():
                DATA[key] = None
            flag = True
        DATA.loc[row[0],res.keys()] = pd.Series(res)

DATA.to_pickle('selkosanomat_news.pickle')


def filewriter(content,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
def filereader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()
def make_comparison_html(clean_htmlA,url_a,clean_htmlB,url_b,output_file):

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
        font-size: 18px;
        margin-bottom: 10px;
    }}
    </style>
    </head>
    <body>
    <div class="container">  
        <div class="left">
            <div class="news-url">
                <a href="{url_a}" class="news-url">{url_a}</a>                
            </div>  
            {contentA}
        </div>
        <div class="right">
            <div class="news-url">
                {url_b}
            </div>          
            {contentB}
        </div>
    </div>
    </body>
    </html>
    '''

    # Fill the template
    html_output = html_template.format(contentA=clean_htmlA, contentB=clean_htmlB,url_a=url_a,url_b=url_b)

    filewriter(html_output, output_file)


def tagged_text_to_colored_html(tagged_text):
    from bs4 import BeautifulSoup, NavigableString, Tag
    from html import escape
    import re

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

OUTPUT_FOLDER = os.getcwd() + os.sep + 'selkosanomat_final'
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER  , exist_ok=True)

good = (DATA['total_text_length']>2500) & (DATA['subtitle_count']>0) & (DATA['subtitle_count']<11)

for row in DATA.loc[good].iterrows():
    textfile = OUTPUT_FOLDER + os.sep + f'selkosanomat_{row[0]}.txt'
    selko = row[1]['tagged_text']
    filewriter(selko,textfile)

    selko = row[1]['tagged_text']
    htmlfile = OUTPUT_FOLDER + os.sep + f'selkosanomat_{row[0]}.html'

    selko_html = tagged_text_to_colored_html(selko)
    regular_html = row[1]['article_html'].replace('.jpg','').replace('.png','')
    make_comparison_html(regular_html,f"Web {row[1]['url']}",selko_html,f'SELKO id {row[0]}',htmlfile)

print('ALL DONE')

