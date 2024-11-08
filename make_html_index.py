import os

# Define your GitHub Pages base URL
base_url = "https://kauttoj.github.io/HH_selkobot"

# Define the folder where your HTML files are located
folder_path = "data/html_selkomedia/formatted"

# Create an index.html file to store the list of links
with open("index.html", "w") as index_file:
    index_file.write("<html><body><h1>HTML Files</h1><ul>")

    # Walk through the folder and find all HTML files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".html"):
                # Create the URL for each HTML file
                relative_path = os.path.join(root, file).replace("\\", "/")
                file_url = f"{base_url}/{relative_path}"
                index_file.write(f'<li><a href="{file_url}">{file}</a></li>\n')

    index_file.write("</ul></body></html>")