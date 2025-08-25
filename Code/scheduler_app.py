import os
import json
import smtplib
import pickle
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Core Project Imports ---
# These functions handle the main logic of the application.
from Clustering import ConvertCurrent
from Vectorize import lade_json
from generate_static_site import generate_static_site

# --- Configuration ---
# It is strongly recommended to use environment variables for sensitive data.
# The GitHub Actions workflow will set these variables from repository secrets.
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
# The receiver email(s) can be hardcoded if they are not sensitive.
RECEIVER_EMAILS = ['Fabian.Acem@kds-software.com']

# This should be the URL where you host your static site (e.g., GitHub Pages).
# Replace <YOUR_USERNAME> and <YOUR_REPONAME> with your actual GitHub details.
WEBSITE_LINK = 'https://<YOUR_USERNAME>.github.io/<YOUR_REPONAME>/'


def run_weekly_workflow():
    """
    Executes the entire weekly workflow:
    1. Loads the ML model.
    2. Fetches and processes new articles.
    3. Makes predictions.
    4. Generates a static HTML website.
    5. Sends an email notification with a link to the site.
    """
    print("--- Starting Weekly Workflow ---")

    # --- 1. Load the ML Model ---
    base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
    model_path = os.path.join(base_data_path, "RF_best_model.pkl")
    try:
        with open(model_path, "rb") as f:
            ml_model = pickle.load(f)
        print("✅ Machine Learning model loaded successfully.")
    except Exception as e:
        print(f"❌ Fatal Error: Could not load the machine learning model from {model_path}. Error: {e}")
        return # Exit if the model isn't available

    # --- 2. Fetch, Process, and Predict ---
    print("\nStep 2: Processing new articles and generating predictions...")
    try:
        # This function handles fetching, processing, and saving the predictions.
        # We pass the pre-loaded model to avoid retraining.
        ConvertCurrent(model=ml_model)
        print("✅ Article processing and prediction complete.")
    except Exception as e:
        print(f"❌ Error during article processing and prediction. Error: {e}")
        # We can still try to generate the site and send an email if predictions exist from a previous run.
    
    # --- 3. Generate Static Website ---
    print("\nStep 3: Generating static HTML website...")
    try:
        generate_static_site()
        print("✅ Static website generated successfully in the 'docs' directory.")
    except Exception as e:
        print(f"❌ Error generating the static site. Error: {e}")
        # If site generation fails, there's no point in sending an email.
        return

    # --- 4. Send Email Notification ---
    print("\nStep 4: Preparing and sending email notification...")
    # Check for email credentials
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("⚠️ Warning: EMAIL_ADDRESS or EMAIL_PASSWORD environment variables not set. Skipping email notification.")
        return

    # Prepare email content
    msg = MIMEMultipart("alternative")
    msg['Subject'] = "Your new personalized article recommendations are here! 🚀"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(RECEIVER_EMAILS)

    # Get the number of suggested articles for the email body
    try:
        suggestions_path = os.path.join(base_data_path, "suggested_articles_for_web.json")
        suggested_ids = lade_json(suggestions_path)
        article_count = len(suggested_ids)
    except Exception:
        article_count = 0

    text = f"Hello,\n\nYour {article_count} new personalized article suggestions are available!\n\nClick the link to view them:\n{WEBSITE_LINK}\n\nEnjoy reading!\n"
    html = f"""    <html>
      <body>
        <p>Hello,</p>
        <p>Your <strong>{article_count}</strong> new personalized article suggestions are available! 🚀</p>
        <p>Click the button below to see your articles:</p>
        <p><a href="{WEBSITE_LINK}" style=\"background-color: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px;\">View My Articles</a></p>
        <p>Enjoy reading!</p>
      </body>
    </html>
    """
    msg.attach(MIMEText(text, 'plain'))
    msg.attach(MIMEText(html, 'html'))

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, RECEIVER_EMAILS, msg.as_string())
        print(f"✅ Email successfully sent to {', '.join(RECEIVER_EMAILS)}!")
    except Exception as e:
        print(f"❌ Fatal Error: Failed to send email. Please check your credentials and SMTP settings. Error: {e}")

    print("\n--- Weekly Workflow Finished ---")

if __name__ == '__main__':
    # This script is designed to be run once by a scheduler like GitHub Actions.
    # It will execute the entire workflow and then exit.
    run_weekly_workflow()
