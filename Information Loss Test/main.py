import json
import re
import os

# ==========================================
# CHANGE THIS ID TO EXTRACT A DIFFERENT STORY
TARGET_ARTICLE_ID = "52845"
# ==========================================

def extract_article(article_id, input_file="QuALITY.v1.0.1.htmlstripped.txt"):
    output_file = f"{article_id}.txt"
    
    # Check if the dataset file exists
    if not os.path.exists(input_file):
        print(f"Error: Could not find the dataset file '{input_file}'")
        return

    found = False

    # Open the dataset and scan line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue # Skip empty lines
            
            # Parse the current line into a JSON object
            item = json.loads(line)
            
            # Check if this line's article_id matches the target
            if item.get('article_id') == article_id:
                found = True
                
                # Extract the relevant fields
                title = item.get('title', 'Unknown Title')
                author = item.get('author', 'Unknown Author')
                raw_text = item.get('article', '')
                
                # Clean up the excessive newlines and transcriber notes
                cleaned_text = re.sub(r'\n\s*\n+', '\n\n', raw_text)
                cleaned_text = re.sub(r'\[Transcriber\'s Note:.*?\]\n+', '', cleaned_text, flags=re.DOTALL)
                
                # Write it to your new text file
                with open(output_file, 'w', encoding='utf-8') as out:
                    out.write(f"Title: {title}\n")
                    out.write(f"Author: {author}\n")
                    out.write(f"Article ID: {article_id}\n")
                    out.write("="*40 + "\n\n")
                    out.write(cleaned_text.strip())
                    
                print(f"Success! '{title}' was found and saved as {output_file}")
                break # Stop searching the file once we find it
                
    if not found:
        print(f"Sorry, an article with the ID '{article_id}' was not found in the dataset.")

# Run the extraction using the variable set at the top
extract_article(TARGET_ARTICLE_ID)