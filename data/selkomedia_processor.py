import glob
import os
from bs4 import BeautifulSoup, NavigableString, Tag
from html import escape
import shutil
import numpy as np
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

SAVE_RESULTS_ON_DISK = 1

FILENAME = 'selkomedia_texts.pickle'
df=pd.read_pickle(FILENAME)

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


def tagged_text_to_colored_html(tagged_text,add_color_code=True):
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

    color_string = ''
    if add_color_code:
        color_string = color_coding_html

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
{color_string}
<p>-------------------------------</p>
{html_text}
</body>
</html>'''

    return html_output

def remove_tags(input_text):
    # Define various quotation marks that might appear
    quotation_marks = ['“', '”','«', '»','‘', '’','❝', '❞','〝', '〞']
    # Function to handle <quote> tags specifically
    def handle_quote(match):
        content = match.group(1).strip()
        # Remove any existing quotation marks
        if any([content[0] in quotation_marks]):
            content = content[1:]
        if any([content[-1] in quotation_marks]):
            content = content[0:-1]
        # Check if the content does not start and end with ASCII double quotes
        content = f'"{content}"'
        return content

    # Handle <quote> tags separately, adding ASCII double quotes if needed
    output_text = re.sub(r'<quote>(.*?)</quote>', handle_quote, input_text)

    # Remove all other tags
    output_text = re.sub(r'<[^>]+>(.*?)</[^>]+>', r'\1', output_text)

    return output_text


import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_with_stats(data,fileout):
    """
    Plots a histogram with markers for the 75th percentile, median, and mean.

    Parameters:
    data (list or array-like): A list or array of numerical values.
    """
    # Calculate statistics
    mean_value = np.mean(data)
    median_value = np.median(data)
    percentile_75 = np.percentile(data, [12.5,87.5])

    # Create the histogram
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=25, alpha=0.7, color='blue', edgecolor='black')

    # Add lines for the mean, median, and 75th percentile
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='-', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.axvline(percentile_75[0], color='purple', linestyle='-.', linewidth=2,
                label=f'12.5th Percentile: {percentile_75[0]:.2f}')
    plt.axvline(percentile_75[1], color='purple', linestyle='-.', linewidth=2,
                label=f'87.5th Percentile: {percentile_75[1]:.2f}')

    # Add legend
    plt.legend()

    # Add title and labels
    plt.title(
        f"Stats (Mean {mean_value:.2f}, median: {median_value:.2f}, 75th prc [{percentile_75[0]:.2f},{percentile_75[1]:.2f}])")
    plt.xlabel("Values")
    plt.ylabel("Frequency")

    # Show the plot
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(fileout,dpi=200)

INPUT_FOLDER = os.getcwd() + os.sep + 'html_selkomedia' + os.sep + 'tagged_texts'
OUTPUT_FOLDER = os.getcwd() + os.sep + os.sep + 'final'

if SAVE_RESULTS_ON_DISK:
    if os.path.exists(OUTPUT_FOLDER):
        # Delete the folder
        shutil.rmtree(OUTPUT_FOLDER)

    os.makedirs(OUTPUT_FOLDER  , exist_ok=True)

files = glob.glob(INPUT_FOLDER + os.sep + '*.txt')
ids = set([x.split('.txt')[0].split('_')[-1] for x in files])

print('found %i ids' % len(ids))
chars = {}
words = {}
for id in ids:
    info = df.loc[df['ID']==id].iloc[0].to_dict()

    files = glob.glob(INPUT_FOLDER + os.sep + f'*_{id}.txt')
    assert len(files)==2,'not enough files!'
    if '_regular_' not in os.path.basename(files[1]):
        files = [files[1]] + [files[0]]
    assert '_regular_' in os.path.basename(files[1]) and '_selko_' in os.path.basename(files[0]),'bad file order!'
    selko = filereader(files[0])
    regular = filereader(files[1])

    chars[id]={'selko':len(selko),'regular':len(regular)}
    words[id] = {'selko': len(word_tokenize(remove_tags(selko), language='finnish')), 'regular': len(word_tokenize(remove_tags(regular), language='finnish'))}

    if SAVE_RESULTS_ON_DISK:
        filewriter(selko,OUTPUT_FOLDER + os.sep + f'{id}_selko.txt')
        filewriter(regular, OUTPUT_FOLDER + os.sep + f'{id}_regular.txt')

    output_file = OUTPUT_FOLDER + os.sep + f'{id}.html'
    selko_html = tagged_text_to_colored_html(selko)
    regular_html = tagged_text_to_colored_html(regular,add_color_code=False)
    if SAVE_RESULTS_ON_DISK:
        make_comparison_html(selko_html,f"SELKO {info['TYPE_A_URL']}",regular_html,f"REGULAR id {info['TYPE_B_URL']}",output_file)

regular = [x['regular'] for _,x in chars.items()]
selko = [x['selko'] for _,x in chars.items()]
print('regular chars = {regular}, selko chars = {selko}, ratio = {ratio}'.format(regular=np.mean(regular),selko=np.mean(selko),ratio=np.mean([selko[k]/regular[k] for k in range(len(selko))])))

plot_histogram_with_stats(regular,OUTPUT_FOLDER + os.sep + 'regular_characters_hist.png')
plot_histogram_with_stats(selko,OUTPUT_FOLDER + os.sep + 'selko_characters_hist.png')

regular = [x['regular'] for _,x in words.items()]
selko = [x['selko'] for _,x in words.items()]
print('regular words = {regular}, selko words = {selko}, ratio = {ratio}'.format(regular=np.mean(regular),selko=np.mean(selko),ratio=np.mean([selko[k]/regular[k] for k in range(len(selko))])))

plot_histogram_with_stats(regular,OUTPUT_FOLDER + os.sep + 'regular_words_hist.png')
plot_histogram_with_stats(selko,OUTPUT_FOLDER + os.sep + 'selko_words_hist.png')

print('all done!')



