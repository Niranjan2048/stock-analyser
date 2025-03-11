import os
import requests
from bs4 import BeautifulSoup
import re
import tkinter as tk
from tkinter import messagebox

# Function to sanitize filenames by removing invalid characters
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to download PDFs from the given URL
def download_pdfs(target_url):
    # Your login credentials (make sure to handle this securely in production)
    username = 'contact@clearviewcapital.in'
    password = '$nowball@123'

    # Create a session to persist cookies
    session = requests.Session()

    # URL of the login page
    login_url = 'https://www.screener.in/login/'  # Replace with actual login URL

    # Step 1: Fetch the login page to get the CSRF token and cookies
    login_page_response = session.get(login_url)
    soup = BeautifulSoup(login_page_response.content, 'html.parser')
    csrf_token = soup.find('input', attrs={'name': 'csrfmiddlewaretoken'})['value']

    # Prepare the payload for login
    payload = {
        'csrfmiddlewaretoken': csrf_token,
        'id_username': username,
        'id_password': password,
        'next': '',
    }

    # Add headers including Referer
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': login_url,
    }

    # Step 3: Log in to the website
    response = session.post(login_url, data=payload, headers=headers)

    if response.ok and "Welcome back" in response.text:
        output_message = "Login successful! Downloading PDFs...\n"

        # Create a folder to save PDFs if it doesn't exist
        if not os.path.exists('pdfs'):
            os.makedirs('pdfs')

        # Access the target page with Referer header set
        response = session.get(target_url, headers={'Referer': login_url})
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links to PDF files
        pdf_links = soup.find_all('a', href=True)

        for link in pdf_links:
            href = link['href']
            if href.endswith('.pdf'):
                pdf_url = href if href.startswith('http') else f'https://www.screener.in{href}'
                pdf_response = session.get(pdf_url, stream=True, headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Referer': target_url,
                })

                if pdf_response.status_code == 200:
                    pdf_name = sanitize_filename(os.path.basename(href))
                    pdf_path = os.path.join('pdfs', pdf_name)

                    with open(pdf_path, 'wb') as pdf_file:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            pdf_file.write(chunk)
                    output_message += f'Downloaded: {pdf_path}\n'
                else:
                    output_message += f'Failed to download {pdf_url}: {pdf_response.status_code}\n'

        messagebox.showinfo("Download Complete", output_message)
    else:
        messagebox.showerror("Login Failed", "Please check your credentials or form data.")

# Function called when the button is clicked
def on_download_button_click():
    target_url = url_entry.get()
    if target_url:
        download_pdfs(target_url)
    else:
        messagebox.showwarning("Input Error", "Please enter a valid URL.")

# Create the main application window
root = tk.Tk()
root.title("PDF Downloader")

# Create an entry widget for the target URL
url_label = tk.Label(root, text="Enter Target URL:")
url_label.pack(pady=10)

url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=10)

# Create a button that triggers the download function
download_button = tk.Button(root, text="Download PDFs", command=on_download_button_click)
download_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
