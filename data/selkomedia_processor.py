import glob
import os
from bs4 import BeautifulSoup, NavigableString, Tag
from html import escape
import shutil
import numpy as np
import re

def filewriter(content,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def filereader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()
def tagged_text_to_colored_html(tagged_text):

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
                {url_a}
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

INPUT_FOLDER = os.getcwd() + os.sep + 'html_selkomedia' + os.sep + 'tagged_texts'
OUTPUT_FOLDER = os.getcwd() + os.sep + os.sep + 'final'

if os.path.exists(OUTPUT_FOLDER):
    # Delete the folder
    shutil.rmtree(OUTPUT_FOLDER)

os.makedirs(OUTPUT_FOLDER  , exist_ok=True)

files = glob.glob(INPUT_FOLDER + os.sep + '*.txt')
ids = set([x.split('.txt')[0].split('_')[-1] for x in files])

print('found %i ids' % len(ids))
chars = {}
for id in ids:
    files = glob.glob(INPUT_FOLDER + os.sep + f'*_{id}.txt')
    assert len(files)==2,'not enough files!'
    if '_regular_' not in os.path.basename(files[1]):
        files = [files[1]] + [files[0]]
    assert '_regular_' in os.path.basename(files[1]) and '_selko_' in os.path.basename(files[0]),'bad file order!'
    selko = filereader(files[0])
    regular = filereader(files[1])

    chars[id]={'selko':len(selko),'regular':len(regular)}

    filewriter(selko,OUTPUT_FOLDER + os.sep + f'{id}_selko.txt')
    filewriter(regular, OUTPUT_FOLDER + os.sep + f'{id}_regular.txt')

    output_file = OUTPUT_FOLDER + os.sep + f'{id}.html'
    selko_html = tagged_text_to_colored_html(selko)
    regular_html = tagged_text_to_colored_html(regular)
    make_comparison_html(selko_html,f'SELKO id {id}',regular_html,f'REGULAR id {id}',output_file)

regular = np.mean([x['regular'] for _,x in chars.items()])
selko = np.mean([x['selko'] for _,x in chars.items()])
print('regular chars = {regular}, selko chars = {selko}, ratio = {ratio}'.format(regular=regular,selko=selko,ratio=selko/regular))

print('all done!')



