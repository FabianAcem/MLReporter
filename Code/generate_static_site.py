import os
import json
from jinja2 import Environment, FileSystemLoader

def generate_static_site():
    """
    Generates a static HTML page from the top recommended articles.
    """
    # --- Path Definitions ---
    # Determine the project's base directory relative to this script's location
    # This script is in Code/, so we go up one level.
    project_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = os.path.join(project_base_dir, 'Data')
    templates_dir = os.path.join(project_base_dir, 'Code', 'templates')
    output_dir = os.path.join(project_base_dir, 'docs')

    # Input file paths
    all_articles_path = os.path.join(data_dir, 'ConvertedCurrent.json')
    suggestions_path = os.path.join(data_dir, 'suggested_articles_for_web.json')

    # Output file path
    output_html_path = os.path.join(output_dir, 'index.html')

    print("Starting static site generation...")
    print(f"Template directory: {templates_dir}")
    print(f"Output directory: {output_dir}")

    # --- Data Loading ---
    try:
        with open(all_articles_path, 'r', encoding='utf-8') as f:
            all_articles = json.load(f)
        with open(suggestions_path, 'r', encoding='utf-8') as f:
            suggested_ids = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse a JSON file. {e}")
        return

    # Create a dictionary for quick lookups of articles by their ID
    articles_by_id = {str(article['id']): article for article in all_articles}

    # Filter to get the top 5 articles based on the suggestions
    top_articles = []
    for article_id in suggested_ids:
        article = articles_by_id.get(str(article_id))
        if article:
            # Ensure the article has a URL, title, and summary
            # The template expects 'url', but the data might have 'link'. Let's check.
            # Assuming the key is 'link' and we map it to 'url' for the template.
            top_articles.append({
                'title': article.get('title', 'No Title'),
                'summary': article.get('summary', 'No summary available.'),
                'source': article.get('source', 'Unknown Source'),
                'url': article.get('link', '#') # Map 'link' to 'url'
            })

    print(f"Found {len(top_articles)} articles to display.")

    # --- Template Rendering ---
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('static_template.html')

    # Render the template with the article data
    rendered_html = template.render(articles=top_articles)

    # --- Output ---
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the rendered HTML to the output file
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
        print(f"Successfully generated static site at: {output_html_path}")
    except IOError as e:
        print(f"Error writing to output file: {e}")

if __name__ == '__main__':
    generate_static_site()
