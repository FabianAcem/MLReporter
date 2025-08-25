import os
import json
from jinja2 import Environment, FileSystemLoader

def generate_static_site():
    """
    Generates a static HTML website from the recommended articles.
    """
    print("--- Starting static site generation ---")

    # --- Configuration ---
    # Define paths relative to this script's location
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(code_dir)
    data_dir = os.path.join(base_dir, 'Data')
    templates_dir = os.path.join(code_dir, 'templates')
    output_dir = os.path.join(base_dir, 'docs')

    # This is the placeholder for the URL of your running Flask app's feedback endpoint.
    # The user of this project will need to replace this with the actual public URL.
    # For example: 'https://your-app-name.herokuapp.com/submit_starred'
    FEEDBACK_URL = os.environ.get('FEEDBACK_APP_URL', 'https://example.com/submit_starred')


    # --- 1. Load Data ---
    try:
        suggested_ids_path = os.path.join(data_dir, "suggested_articles_for_web.json")
        with open(suggested_ids_path, 'r', encoding='utf-8') as f:
            suggested_articles_ids = json.load(f)

        all_articles_path = os.path.join(data_dir, "ConvertedCurrent.json")
        with open(all_articles_path, 'r', encoding='utf-8') as f:
            all_current_articles = json.load(f)

        articles_to_display = [article for article in all_current_articles if article.get("id") in suggested_articles_ids]
        print(f"✅ Loaded {len(articles_to_display)} articles to display.")

    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found - {e}. Aborting site generation.")
        articles_to_display = []
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Could not decode JSON from data file - {e}. Aborting site generation.")
        return

    # --- 2. Set up Jinja2 Environment ---
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('index.html')

    # --- 3. Render the Template ---
    # We need a modified version of the template for static generation.
    # For now, we will pass the feedback URL to the template context.
    # In a later step, we will modify the template itself.
    html_content = template.render(
        articles=articles_to_display,
        submit_url=FEEDBACK_URL
    )
    print("✅ HTML content rendered successfully.")

    # --- 4. Write to Output File ---
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Created output directory: {output_dir}")

        output_path = os.path.join(output_dir, 'index.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ Static site successfully generated at: {output_path}")

    except Exception as e:
        print(f"❌ Error writing the static HTML file: {e}")

    print("--- Static site generation finished ---")

if __name__ == '__main__':
    # This allows running the script directly for testing purposes.
    generate_static_site()
