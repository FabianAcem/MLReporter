# Weekly Article Selection Workflow

This repository contains the code for an automated weekly workflow that selects and delivers the five most interesting articles from Inoreader using a machine learning model.

### Required GitHub Secrets

The following environment variables must be configured as **secrets** in the GitHub repository settings. These are necessary for the workflow to access external services securely.

-   **INOREADER_CLIENT_ID**: The client ID for the Inoreader API application.
-   **INOREADER_CLIENT_SECRET**: The client secret for the Inoreader API application.
-   **INOREADER_REFRESH_TOKEN**: A refresh token to generate new access tokens for Inoreader.
-   **INOREADER_APP_ID**: The App ID for the Inoreader API application.
-   **INOREADER_APP_KEY**: The App Key for the Inoreader API application.
-   **EMAIL_ADDRESS**: The sender's email address (e.g., `florian@meinedomain.de`).
-   **FEEDBACK_APP_URL**: The public URL where the feedback server (`app.py`) is hosted.
-   **STATIC_SITE_URL**: The URL where the static articles page is hosted.

### Email Configuration

This workflow uses a specific SMTP server that does **not** require authentication. The following details are hardcoded within the script and do not need to be stored as secrets:

-   **SMTP Host:** `prastaro2.praha.bcpraha.com`
-   **SMTP Port:** `25`
-   **Authentication:** `None`

---

Nachdem Sie den Text in die `README.md` eingefügt haben, speichern Sie die Datei, committen Sie die Änderung (`git add .`, `git commit -m "Updated README with secrets and SMTP details"`) und pushen Sie sie auf Ihr GitHub-Repository (`git push`).
