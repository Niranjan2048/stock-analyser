import os
import requests
from bs4 import BeautifulSoup
import re
import json
import logging
import time
from datetime import datetime, timedelta
import traceback
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_document_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("financial_document_collector")


class FinancialDocumentCollector:
    """Collects annual reports and concall documents from Screener.in for any company"""
    
    def __init__(self, config_path="config.json", company_config=None):
        """Initialize with configuration
        
        Args:
            config_path (str): Path to the config file (used if company_config is None)
            company_config (dict): Direct configuration dictionary (overrides config_path)
        """
        if company_config:
            self.config = company_config
            self.config_path = None
        else:
            self.config = self._load_config(config_path)
            self.config_path = config_path
            
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Validate configuration
        if not self._validate_config():
            raise ValueError("Invalid configuration. Please check config.json")
        
        # Extract company information
        self.company_id = self.config["company"]["id"]
        self.company_name = self.config["company"]["name"]
        self.company_url = self.config["company"]["screener_url"]
        
        # Create base directory
        self.base_dir = os.path.join(
            self.config.get("base_dir", "financial_data"),
            self.sanitize_filename(self.company_id)
        )
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        logger.info(f"Data will be saved to {self.base_dir}")
            
        # Create subdirectories for different document types
        self.annual_reports_dir = os.path.join(self.base_dir, "annual_reports")
        os.makedirs(self.annual_reports_dir, exist_ok=True)
        
        self.concalls_dir = os.path.join(self.base_dir, "concalls")
        os.makedirs(self.concalls_dir, exist_ok=True)
        
        # Create directories for different concall document types
        for doc_type in ["transcript", "notes", "ppt"]:
            os.makedirs(os.path.join(self.concalls_dir, doc_type), exist_ok=True)
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Creating default configuration.")
            default_config = {
                "company": {
                    "id": "HDFCBANK",
                    "name": "HDFC Bank Ltd",
                    "screener_url": "https://www.screener.in/company/HDFCBANK"
                },
                "screener_credentials": {
                    "username": "",
                    "password": ""
                },
                "base_dir": "financial_data",
                "annual_reports_years": 5,  # Last 5 years of annual reports
                "concalls_months": 6,       # Last 6 months of concalls
                "max_retries": 3,
                "retry_delay": 5,
                "debug_mode": True          # Save HTML for debugging
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _validate_config(self):
        """Validate the configuration"""
        # Check company information
        if "company" not in self.config:
            logger.error("Missing company configuration")
            return False
        
        required_fields = ["id", "name", "screener_url"]
        for field in required_fields:
            if field not in self.config["company"]:
                logger.error(f"Missing company.{field} configuration")
                return False
        
        return True
    
    def sanitize_filename(self, filename):
        """Sanitize filenames by removing invalid characters"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    def login(self):
        """Login to Screener.in"""
        login_url = 'https://www.screener.in/login/'
        credentials = self.config.get("screener_credentials", {})
        
        # Skip login if credentials are not provided
        if not credentials.get("username") or not credentials.get("password"):
            logger.warning("Screener.in credentials not configured - proceeding without login")
            return True  # Return True to allow script to continue
        
        try:
            # Get CSRF token
            login_page = self.session.get(login_url)
            soup = BeautifulSoup(login_page.content, 'html.parser')
            csrf_token_element = soup.find('input', attrs={'name': 'csrfmiddlewaretoken'})
            
            if not csrf_token_element:
                logger.error("Could not find CSRF token for login")
                if self.config.get("debug_mode", False):
                    with open(os.path.join(self.base_dir, "login_page.html"), "w", encoding='utf-8') as f:
                        f.write(login_page.text)
                return False
            
            csrf_token = csrf_token_element['value']
            
            # Login
            payload = {
                'csrfmiddlewaretoken': csrf_token,
                'id_username': credentials["username"],
                'id_password': credentials["password"],
                'next': '',
            }
            
            headers = {
                'Referer': login_url,
            }
            
            response = self.session.post(login_url, data=payload, headers=headers)
            
            if response.ok and "Welcome back" in response.text:
                logger.info("Successfully logged in to Screener.in")
                return True
            else:
                logger.error(f"Failed to login to Screener.in: {response.status_code}")
                if self.config.get("debug_mode", False):
                    with open(os.path.join(self.base_dir, "login_response.html"), "w", encoding='utf-8') as f:
                        f.write(response.text)
                return False
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def download_document(self, url, doc_type, sub_type=None, period=None):
        """Download a document from URL"""
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay", 5)
        
        if not url or not url.strip():
            logger.warning(f"Empty URL provided for {doc_type}/{sub_type} - {period}")
            return None
            
        # Skip URLs that don't point to actual documents
        if "javascript:void" in url or "#" == url:
            logger.warning(f"Skipping non-document URL: {url}")
            return None
        
        for attempt in range(max_retries):
            try:
                # Prepare directory and file name
                if doc_type == "annual_reports":
                    dir_path = self.annual_reports_dir
                elif doc_type == "concalls" and sub_type:
                    dir_path = os.path.join(self.concalls_dir, sub_type.lower())
                else:
                    dir_path = os.path.join(self.base_dir, doc_type)
                
                # Determine filename
                url_path = url.split("?")[0]  # Remove query parameters
                original_filename = os.path.basename(url_path)
                
                # Handle empty filenames or URLs ending with a slash
                if not original_filename:
                    original_filename = f"document_{int(time.time())}"
                
                if period:
                    # Include period in filename for better organization
                    filename = f"{period}_{self.sanitize_filename(original_filename)}"
                else:
                    filename = self.sanitize_filename(original_filename)
                
                # If URL doesn't end with a file extension, append .pdf for PDFs
                if '.' not in filename and 'pdf' in url.lower():
                    filename += '.pdf'
                elif '.' not in filename and 'youtube.com' in url.lower():
                    filename += '.url'  # YouTube links are saved as .url files
                
                file_path = os.path.join(dir_path, filename)
                
                # Check if file already exists
                if os.path.exists(file_path):
                    logger.info(f"File already exists: {file_path}")
                    return file_path
                
                # Download the file
                logger.info(f"Downloading: {url}")
                
                # For YouTube links, just create a .url file
                if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                    with open(file_path, 'w') as f:
                        f.write(f"[InternetShortcut]\nURL={url}")
                    logger.info(f"Created URL shortcut: {file_path}")
                    return file_path
                
                # Send a HEAD request first to check if the URL is accessible
                try:
                    head_response = self.session.head(url, allow_redirects=True, timeout=30)
                    if not head_response.ok:
                        logger.warning(f"URL not accessible: {url} - Status: {head_response.status_code}")
                        time.sleep(retry_delay)
                        continue
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error checking URL {url}: {e}")
                    time.sleep(retry_delay)
                    continue
                
                # Download the file
                try:
                    response = self.session.get(url, stream=True, timeout=60, 
                                            headers={'Referer': 'https://www.screener.in/'})
                    
                    if not response.ok:
                        logger.warning(f"Failed to download {url}: {response.status_code}")
                        time.sleep(retry_delay)
                        continue
                    
                    # Save the file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"Successfully downloaded: {file_path}")
                    return file_path
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error downloading {url}: {e}")
                    time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(retry_delay)
        
        logger.error(f"Failed to download {url} after {max_retries} attempts")
        return None
    
    def extract_concall_notes(self, note_url, period):
        """Extract concall notes from a modal dialog and save to file"""
        try:
            # Extract the concall ID from the URL
            concall_id_match = re.search(r'summary/(\d+)/', note_url)
            if not concall_id_match:
                logger.warning(f"Could not extract concall ID from URL: {note_url}")
                return None
            
            concall_id = concall_id_match.group(1)
            
            # Make a request to get the notes content
            notes_url = f"https://www.screener.in/concalls/summary/{concall_id}/"
            response = self.session.get(notes_url)
            
            if not response.ok:
                logger.warning(f"Failed to get concall notes from {notes_url}: {response.status_code}")
                return None
            
            # Parse the HTML response
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the content inside the modal
            modal_content = soup.select_one('.modal-content')
            if not modal_content:
                logger.warning(f"Could not find modal content in response from {notes_url}")
                return None
            
            # Extract the text content
            notes_content = modal_content.get_text(separator='\n', strip=True)
            
            # Clean up the content (remove any start/end markers like ```)
            notes_content = re.sub(r'^```|```$', '', notes_content).strip()
            
            # Save to file
            dir_path = os.path.join(self.concalls_dir, "notes")
            file_path = os.path.join(dir_path, f"{period}_notes.md")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(notes_content)
            
            logger.info(f"Successfully extracted and saved concall notes to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error extracting concall notes: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def parse_annual_reports(self, soup):
        """Parse annual reports from the document page"""
        try:
            years_to_get = self.config.get("annual_reports_years", 5)
            logger.info(f"Parsing last {years_to_get} annual reports")
            
            annual_reports = []
            
            # Try multiple methods to find annual reports
            
            # Method 1: Look for the annual reports list in a dedicated section
            reports_section = soup.select_one('.documents.annual-reports, .annual-reports')
            if reports_section:
                reports_list = reports_section.select('.list-links li a')
                
                for link in reports_list:
                    if 'Financial Year' in link.text:
                        year_match = re.search(r'Financial Year (\d{4})', link.text)
                        if year_match:
                            year = year_match.group(1)
                            href = link.get('href')
                            source = 'bse' if 'bse' in link.text.lower() else 'nse'
                            
                            if href:
                                annual_reports.append({
                                    'year': year,
                                    'url': href,
                                    'source': source
                                })
            
            # Method 2: Look for any links with "Annual Report" or "Financial Year" text
            if not annual_reports:
                logger.info("Trying alternative method to find annual reports")
                for link in soup.select('a'):
                    link_text = link.text.strip()
                    href = link.get('href')
                    
                    if not href:
                        continue
                    
                    # Check for annual report keywords
                    if 'Annual Report' in link_text or 'Financial Year' in link_text:
                        year_match = re.search(r'(?:Financial Year|FY|Annual Report|AR)[^0-9]*(\d{4})', link_text)
                        if year_match:
                            year = year_match.group(1)
                            source = 'bse' if 'bse' in link_text.lower() else 'nse'
                            
                            annual_reports.append({
                                'year': year,
                                'url': href,
                                'source': source
                            })
            
            # Sort by year (newest first) and take only the required number
            annual_reports.sort(key=lambda x: x['year'], reverse=True)
            annual_reports = annual_reports[:years_to_get]
            
            logger.info(f"Found {len(annual_reports)} annual reports")
            return annual_reports
            
        except Exception as e:
            logger.error(f"Error parsing annual reports: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def parse_concalls(self, soup):
        """Parse concalls from the document page with improved selectors and extraction logic"""
        try:
            months_to_get = self.config.get("concalls_months", 6)
            logger.info(f"Parsing concalls from the last {months_to_get} months")
            
            # Calculate the cutoff date
            cutoff_date = datetime.now() - timedelta(days=30 * months_to_get)
            concalls = []
            processed_dates = set()  # To avoid duplicates
            
            # Debug: Save full document HTML
            if self.config.get("debug_mode", False):
                debug_path = os.path.join(self.base_dir, "full_document_page.html")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                logger.info(f"Saved full HTML to {debug_path} for debugging")
                
                # Also save just the concalls section if we can find it
                concalls_section = soup.select_one('#documents .concalls, .documents.concalls')
                if concalls_section:
                    section_path = os.path.join(self.base_dir, "concalls_section.html")
                    with open(section_path, 'w', encoding='utf-8') as f:
                        f.write(str(concalls_section))
                    logger.info(f"Saved concalls section to {section_path} for debugging")
            
            # Setup regex for finding dates
            date_pattern = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}')
            
            # APPROACH 1: Find all elements that directly contain both a date and concall links
            # This approach looks for rows/elements that contain both date and document links
            for element in soup.find_all(['div', 'li', 'tr']):
                # Skip elements that are too large to be a single row
                if len(element.get_text()) > 200:
                    continue
                    
                element_text = element.get_text()
                date_match = date_pattern.search(element_text)
                
                if date_match and ('Transcript' in element_text or 'Notes' in element_text or 'PPT' in element_text):
                    # Extract date
                    date_text = date_match.group(0)
                    month, year = date_text.split()
                    
                    # Skip if already processed this date
                    if date_text in processed_dates:
                        continue
                    processed_dates.add(date_text)
                    
                    try:
                        call_date = datetime.strptime(f"{month} {year}", "%b %Y")
                        
                        # Skip if older than cutoff date
                        if call_date < cutoff_date:
                            continue
                        
                        # Extract document links
                        documents = []
                        
                        # Find transcript link
                        transcript_links = element.find_all('a', string=lambda s: s and 'Transcript' in s)
                        for link in transcript_links:
                            href = link.get('href')
                            if href:
                                if not href.startswith(('http://', 'https://')):
                                    href = urljoin('https://www.screener.in', href)
                                
                                documents.append({
                                    'type': 'transcript',
                                    'url': href,
                                    'is_modal': False
                                })
                        
                        # Find notes button
                        notes_btns = element.find_all('button', string=lambda s: s and 'Notes' in s)
                        for btn in notes_btns:
                            data_url = btn.get('data-url')
                            if data_url:
                                if not data_url.startswith(('http://', 'https://')):
                                    data_url = urljoin('https://www.screener.in', data_url)
                                
                                documents.append({
                                    'type': 'notes',
                                    'url': data_url,
                                    'is_modal': True
                                })
                        
                        # Find PPT link
                        ppt_links = element.find_all('a', string=lambda s: s and 'PPT' in s)
                        for link in ppt_links:
                            href = link.get('href')
                            if href:
                                if not href.startswith(('http://', 'https://')):
                                    href = urljoin('https://www.screener.in', href)
                                
                                documents.append({
                                    'type': 'ppt',
                                    'url': href,
                                    'is_modal': False
                                })
                        
                        # If we found documents, add this concall
                        if documents:
                            concalls.append({
                                'date': call_date,
                                'period': f"{month}_{year}",
                                'documents': documents
                            })
                            logger.info(f"Found concall for {month} {year} with {len(documents)} documents (approach 1)")
                    except Exception as e:
                        logger.error(f"Error processing concall date {date_text} in approach 1: {e}")
                        logger.error(traceback.format_exc())
            
            # APPROACH 2: Find date elements, then look for adjacent document links
            # This handles layouts where date and links are in separate but adjacent elements
            if len(concalls) < months_to_get:
                logger.info("Using approach 2: Find date elements then adjacent documents")
                
                # Find all elements that look like pure date elements
                date_elements = []
                for element in soup.find_all(text=date_pattern):
                    parent = element.parent
                    if parent and parent.get_text().strip() == element.strip():
                        date_elements.append(parent)
                
                logger.info(f"Found {len(date_elements)} potential pure date elements")
                
                for date_elem in date_elements:
                    date_text = date_elem.get_text().strip()
                    date_match = date_pattern.search(date_text)
                    
                    if date_match:
                        date_text = date_match.group(0)
                        month, year = date_text.split()
                        
                        # Skip if already processed
                        if date_text in processed_dates:
                            continue
                        processed_dates.add(date_text)
                        
                        try:
                            call_date = datetime.strptime(f"{month} {year}", "%b %Y")
                            
                            # Skip if older than cutoff date
                            if call_date < cutoff_date:
                                continue
                            
                            # Look for document links in nearby elements
                            documents = []
                            found_docs = False
                            
                            # Check next sibling
                            next_elem = date_elem.next_sibling
                            while next_elem and isinstance(next_elem, str) and next_elem.strip() == '':
                                next_elem = next_elem.next_sibling
                            
                            if next_elem and hasattr(next_elem, 'find_all'):
                                # Transcript link in next sibling
                                for link in next_elem.find_all('a', string=lambda s: s and 'Transcript' in s):
                                    href = link.get('href')
                                    if href:
                                        if not href.startswith(('http://', 'https://')):
                                            href = urljoin('https://www.screener.in', href)
                                        
                                        documents.append({
                                            'type': 'transcript',
                                            'url': href,
                                            'is_modal': False
                                        })
                                        found_docs = True
                                
                                # Notes button in next sibling
                                for btn in next_elem.find_all('button', string=lambda s: s and 'Notes' in s):
                                    data_url = btn.get('data-url')
                                    if data_url:
                                        if not data_url.startswith(('http://', 'https://')):
                                            data_url = urljoin('https://www.screener.in', data_url)
                                        
                                        documents.append({
                                            'type': 'notes',
                                            'url': data_url,
                                            'is_modal': True
                                        })
                                        found_docs = True
                                
                                # PPT link in next sibling
                                for link in next_elem.find_all('a', string=lambda s: s and 'PPT' in s):
                                    href = link.get('href')
                                    if href:
                                        if not href.startswith(('http://', 'https://')):
                                            href = urljoin('https://www.screener.in', href)
                                        
                                        documents.append({
                                            'type': 'ppt',
                                            'url': href,
                                            'is_modal': False
                                        })
                                        found_docs = True
                            
                            # If we didn't find docs in next sibling, try parent's next sibling
                            if not found_docs and date_elem.parent:
                                parent_next = date_elem.parent.next_sibling
                                while parent_next and isinstance(parent_next, str) and parent_next.strip() == '':
                                    parent_next = parent_next.next_sibling
                                
                                if parent_next and hasattr(parent_next, 'find_all'):
                                    # Transcript link in parent's next sibling
                                    for link in parent_next.find_all('a', string=lambda s: s and 'Transcript' in s):
                                        href = link.get('href')
                                        if href:
                                            if not href.startswith(('http://', 'https://')):
                                                href = urljoin('https://www.screener.in', href)
                                            
                                            documents.append({
                                                'type': 'transcript',
                                                'url': href,
                                                'is_modal': False
                                            })
                                    
                                    # Notes button in parent's next sibling
                                    for btn in parent_next.find_all('button', string=lambda s: s and 'Notes' in s):
                                        data_url = btn.get('data-url')
                                        if data_url:
                                            if not data_url.startswith(('http://', 'https://')):
                                                data_url = urljoin('https://www.screener.in', data_url)
                                            
                                            documents.append({
                                                'type': 'notes',
                                                'url': data_url,
                                                'is_modal': True
                                            })
                                    
                                    # PPT link in parent's next sibling
                                    for link in parent_next.find_all('a', string=lambda s: s and 'PPT' in s):
                                        href = link.get('href')
                                        if href:
                                            if not href.startswith(('http://', 'https://')):
                                                href = urljoin('https://www.screener.in', href)
                                            
                                            documents.append({
                                                'type': 'ppt',
                                                'url': href,
                                                'is_modal': False
                                            })
                            
                            # If we found any documents, add this concall
                            if documents:
                                concalls.append({
                                    'date': call_date,
                                    'period': f"{month}_{year}",
                                    'documents': documents
                                })
                                logger.info(f"Found concall for {month} {year} with {len(documents)} documents (approach 2)")
                        except Exception as e:
                            logger.error(f"Error processing concall date {date_text} in approach 2: {e}")
                            logger.error(traceback.format_exc())
            
            # APPROACH 3: Find transcript links first, then look for nearby dates
            # This works in cases where the transcript link is the most reliable element
            if len(concalls) < months_to_get:
                logger.info("Using approach 3: Find transcript links first, then dates")
                
                # Find all transcript links
                transcript_links = soup.find_all('a', string=lambda s: s and 'Transcript' in s)
                logger.info(f"Found {len(transcript_links)} transcript links")
                
                for link in transcript_links:
                    # Skip if we already have enough concalls
                    if len(concalls) >= months_to_get and len(processed_dates) >= months_to_get:
                        break
                        
                    # Look for date near this link
                    parent = link.parent
                    if not parent:
                        continue
                    
                    # Expand search area: check parent and its parent
                    search_area = parent.get_text()
                    if parent.parent:
                        search_area += ' ' + parent.parent.get_text()
                    
                    # Look for date in the search area
                    date_match = date_pattern.search(search_area)
                    if date_match:
                        date_text = date_match.group(0)
                        month, year = date_text.split()
                        
                        # Skip if already processed
                        if date_text in processed_dates:
                            continue
                        processed_dates.add(date_text)
                        
                        try:
                            call_date = datetime.strptime(f"{month} {year}", "%b %Y")
                            
                            # Skip if older than cutoff date
                            if call_date < cutoff_date:
                                continue
                            
                            # Start building document list with this transcript
                            documents = []
                            href = link.get('href')
                            if href:
                                if not href.startswith(('http://', 'https://')):
                                    href = urljoin('https://www.screener.in', href)
                                
                                documents.append({
                                    'type': 'transcript',
                                    'url': href,
                                    'is_modal': False
                                })
                            
                            # Look for related documents in the same container
                            search_container = parent
                            
                            # If the parent is very small, use its parent instead
                            if len(parent.get_text()) < 20 and parent.parent:
                                search_container = parent.parent
                            
                            # Find notes button
                            notes_btn = search_container.find('button', string=lambda s: s and 'Notes' in s)
                            if notes_btn:
                                data_url = notes_btn.get('data-url')
                                if data_url:
                                    if not data_url.startswith(('http://', 'https://')):
                                        data_url = urljoin('https://www.screener.in', data_url)
                                    
                                    documents.append({
                                        'type': 'notes',
                                        'url': data_url,
                                        'is_modal': True
                                    })
                            
                            # Find PPT link
                            ppt_link = search_container.find('a', string=lambda s: s and 'PPT' in s)
                            if ppt_link:
                                href = ppt_link.get('href')
                                if href:
                                    if not href.startswith(('http://', 'https://')):
                                        href = urljoin('https://www.screener.in', href)
                                    
                                    documents.append({
                                        'type': 'ppt',
                                        'url': href,
                                        'is_modal': False
                                    })
                            
                            # If we found any documents, add this concall
                            if documents:
                                concalls.append({
                                    'date': call_date,
                                    'period': f"{month}_{year}",
                                    'documents': documents
                                })
                                logger.info(f"Found concall for {month} {year} with {len(documents)} documents (approach 3)")
                        except Exception as e:
                            logger.error(f"Error processing concall date from transcript link: {e}")
                            logger.error(traceback.format_exc())
            
            # APPROACH 4: Try to find tables containing concall information
            if len(concalls) < months_to_get:
                logger.info("Using approach 4: Look for tables containing concall information")
                
                # Find all tables
                tables = soup.find_all('table')
                for table in tables:
                    table_text = table.get_text()
                    if 'Transcript' in table_text or 'Notes' in table_text or 'PPT' in table_text:
                        # This table might contain concall info
                        for row in table.find_all('tr'):
                            row_text = row.get_text()
                            date_match = date_pattern.search(row_text)
                            
                            if date_match and ('Transcript' in row_text or 'Notes' in row_text or 'PPT' in row_text):
                                date_text = date_match.group(0)
                                month, year = date_text.split()
                                
                                # Skip if already processed
                                if date_text in processed_dates:
                                    continue
                                processed_dates.add(date_text)
                                
                                try:
                                    call_date = datetime.strptime(f"{month} {year}", "%b %Y")
                                    
                                    # Skip if older than cutoff date
                                    if call_date < cutoff_date:
                                        continue
                                    
                                    # Extract document links
                                    documents = []
                                    
                                    # Transcript links
                                    for link in row.find_all('a', string=lambda s: s and 'Transcript' in s):
                                        href = link.get('href')
                                        if href:
                                            if not href.startswith(('http://', 'https://')):
                                                href = urljoin('https://www.screener.in', href)
                                            
                                            documents.append({
                                                'type': 'transcript',
                                                'url': href,
                                                'is_modal': False
                                            })
                                    
                                    # Notes buttons
                                    for btn in row.find_all('button', string=lambda s: s and 'Notes' in s):
                                        data_url = btn.get('data-url')
                                        if data_url:
                                            if not data_url.startswith(('http://', 'https://')):
                                                data_url = urljoin('https://www.screener.in', data_url)
                                            
                                            documents.append({
                                                'type': 'notes',
                                                'url': data_url,
                                                'is_modal': True
                                            })
                                    
                                    # PPT links
                                    for link in row.find_all('a', string=lambda s: s and 'PPT' in s):
                                        href = link.get('href')
                                        if href:
                                            if not href.startswith(('http://', 'https://')):
                                                href = urljoin('https://www.screener.in', href)
                                            
                                            documents.append({
                                                'type': 'ppt',
                                                'url': href,
                                                'is_modal': False
                                            })
                                    
                                    # If we found any documents, add this concall
                                    if documents:
                                        concalls.append({
                                            'date': call_date,
                                            'period': f"{month}_{year}",
                                            'documents': documents
                                        })
                                        logger.info(f"Found concall for {month} {year} with {len(documents)} documents (approach 4)")
                                except Exception as e:
                                    logger.error(f"Error processing concall date from table: {e}")
                                    logger.error(traceback.format_exc())
            
            # Sort by date (newest first) and remove duplicates
            concalls.sort(key=lambda x: x['date'], reverse=True)
            
            # Remove duplicates based on period
            unique_concalls = []
            seen_periods = set()
            for concall in concalls:
                if concall['period'] not in seen_periods:
                    seen_periods.add(concall['period'])
                    unique_concalls.append(concall)
            
            logger.info(f"Found {len(unique_concalls)} unique concalls within the last {months_to_get} months")
            return unique_concalls
        except Exception as e:
            logger.error(f"Error in concalls parsing: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    def fetch_and_parse_document_page(self):
        """Fetch the document page and parse annual reports and concalls"""
        try:
            # Navigate to the documents page
            documents_url = f"{self.company_url}/consolidated/#documents"
            logger.info(f"Fetching document page: {documents_url}")
            
            response = self.session.get(documents_url)
            
            if not response.ok:
                logger.error(f"Failed to access documents page: {response.status_code}")
                # Try without consolidated view
                documents_url = f"{self.company_url}/#documents"
                logger.info(f"Retrying with: {documents_url}")
                response = self.session.get(documents_url)
                
                if not response.ok:
                    logger.error(f"Failed to access documents page again: {response.status_code}")
                    return None
            
            # Save HTML for debugging if needed
            if self.config.get("debug_mode", False):
                debug_path = os.path.join(self.base_dir, "documents_page.html")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Saved HTML to {debug_path} for debugging")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if we're on the right page
            documents_section = soup.select_one('#documents')
            if not documents_section:
                logger.error("Documents section not found on page")
                return None
            
            return soup
        except Exception as e:
            logger.error(f"Error fetching document page: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def run(self):
        """Run the document collection process"""
        logger.info(f"Starting document collection for {self.company_name} ({self.company_id})")
        
        # Step 1: Login to Screener.in
        login_success = self.login()
        if not login_success:
            logger.warning("Login failed. Attempting to continue without login.")
        
        # Step 2: Fetch and parse the document page
        soup = self.fetch_and_parse_document_page()
        if not soup:
            logger.error("Failed to fetch document page. Aborting collection.")
            return
        
        # Step 3: Parse and download annual reports
        annual_reports = self.parse_annual_reports(soup)
        downloaded_reports = []
        
        for report in annual_reports:
            file_path = self.download_document(
                report['url'], 
                "annual_reports", 
                period=report['year']
            )
            if file_path:
                downloaded_reports.append({
                    'year': report['year'],
                    'source': report.get('source', 'unknown'),
                    'file_path': file_path
                })
        
        logger.info(f"Downloaded {len(downloaded_reports)} annual reports")
        
        # Step 4: Parse and download concalls
        concalls = self.parse_concalls(soup)
        downloaded_concalls = []
        
        for concall in concalls:
            period = concall['period']
            concall_docs = []
            
            for doc in concall['documents']:
                if doc.get('is_modal', False):
                    # This is a modal document (likely notes)
                    file_path = self.extract_concall_notes(doc['url'], period)
                else:
                    # Regular download for non-modal documents
                    file_path = self.download_document(
                        doc['url'], 
                        "concalls", 
                        sub_type=doc['type'],
                        period=period
                    )
                
                if file_path:
                    concall_docs.append({
                        'type': doc['type'],
                        'file_path': file_path
                    })
            
            if concall_docs:
                downloaded_concalls.append({
                    'period': period,
                    'documents': concall_docs
                })
        
        logger.info(f"Downloaded {len(downloaded_concalls)} concall document sets")
        
        # Step 5: Save the results
        results = {
            "company": {
                "id": self.company_id,
                "name": self.company_name,
                "url": self.company_url
            },
            "annual_reports": downloaded_reports,
            "concalls": downloaded_concalls,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.base_dir, "collection_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Document collection for {self.company_name} completed successfully")
        return results


class BatchFinancialDocumentCollector:
    """Collects financial documents for multiple companies"""
    
    def __init__(self, config_path="config.json"):
        """Initialize with configuration"""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Validate the configuration
        if not self._validate_config():
            raise ValueError("Invalid configuration. Please check your config.json")
        
        # Create base directory for all companies
        self.base_dir = self.config.get("base_dir", "financial_data")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # List of companies to process
        self.companies = self.config.get("companies", [])
        
        # Set up shared session for better performance
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check if this is a legacy single-company config
            if "company" in config and not "companies" in config:
                # Convert to multi-company format
                config["companies"] = [config["company"]]
                logger.info("Converted legacy single-company config to multi-company format")
            
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Creating default configuration.")
            default_config = {
                "companies": [
                    {
                        "id": "HDFCBANK",
                        "name": "HDFC Bank Ltd",
                        "screener_url": "https://www.screener.in/company/HDFCBANK"
                    },
                    {
                        "id": "SBIN",
                        "name": "State Bank of India",
                        "screener_url": "https://www.screener.in/company/SBIN"
                    }
                ],
                "screener_credentials": {
                    "username": "",
                    "password": ""
                },
                "base_dir": "financial_data",
                "annual_reports_years": 5,  # Last 5 years of annual reports
                "concalls_months": 6,       # Last 6 months of concalls
                "max_retries": 3,
                "retry_delay": 5,
                "debug_mode": True          # Save HTML for debugging
            }
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _validate_config(self):
        """Validate the configuration"""
        # Check for 'companies' list
        if "companies" not in self.config or not isinstance(self.config["companies"], list) or len(self.config["companies"]) == 0:
            logger.error("Missing or invalid 'companies' configuration (should be a non-empty list)")
            return False
        
        # Check each company entry
        required_fields = ["id", "name", "screener_url"]
        for idx, company in enumerate(self.config["companies"]):
            for field in required_fields:
                if field not in company:
                    logger.error(f"Missing {field} for company at index {idx}")
                    return False
        
        return True
    
    def login(self):
        """Login to Screener.in with shared session"""
        login_url = 'https://www.screener.in/login/'
        credentials = self.config.get("screener_credentials", {})
        
        # Skip login if credentials are not provided
        if not credentials.get("username") or not credentials.get("password"):
            logger.warning("Screener.in credentials not configured - proceeding without login")
            return True  # Return True to allow script to continue
        
        try:
            # Get CSRF token
            login_page = self.session.get(login_url)
            soup = BeautifulSoup(login_page.content, 'html.parser')
            csrf_token_element = soup.find('input', attrs={'name': 'csrfmiddlewaretoken'})
            
            if not csrf_token_element:
                logger.error("Could not find CSRF token for login")
                # Create a debug directory if it doesn't exist
                debug_dir = os.path.join(self.base_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                with open(os.path.join(debug_dir, "login_page.html"), "w", encoding='utf-8') as f:
                    f.write(login_page.text)
                return False
            
            csrf_token = csrf_token_element['value']
            
            # Login
            payload = {
                'csrfmiddlewaretoken': csrf_token,
                'id_username': credentials["username"],
                'id_password': credentials["password"],
                'next': '',
            }
            
            headers = {
                'Referer': login_url,
            }
            
            response = self.session.post(login_url, data=payload, headers=headers)
            
            if response.ok and "Welcome back" in response.text:
                logger.info("Successfully logged in to Screener.in")
                return True
            else:
                logger.error(f"Failed to login to Screener.in: {response.status_code}")
                debug_dir = os.path.join(self.base_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                with open(os.path.join(debug_dir, "login_response.html"), "w", encoding='utf-8') as f:
                    f.write(response.text)
                return False
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run(self):
        """Run the document collection process for all companies"""
        logger.info(f"Starting batch document collection for {len(self.companies)} companies")
        
        # Step 1: Login once to Screener.in (shared for all companies)
        login_success = self.login()
        if not login_success:
            logger.warning("Batch login failed. Individual companies will attempt to login again if needed.")
        
        # Step 2: Process each company
        all_results = []
        
        for idx, company in enumerate(self.companies):
            logger.info(f"Processing company {idx + 1}/{len(self.companies)}: {company['name']} ({company['id']})")
            print(f"\nProcessing company {idx + 1}/{len(self.companies)}: {company['name']} ({company['id']})")
            
            try:
                # Create a configuration for this company with all global settings
                company_config = self.config.copy()
                company_config["company"] = company
                
                # Remove the 'companies' key to avoid confusion
                if "companies" in company_config:
                    del company_config["companies"]
                
                # Process this company
                collector = FinancialDocumentCollector(company_config=company_config)
                
                # Check if we are authenticated, if not, try to login again
                if not login_success:
                    collector.login()
                else:
                    # Use the shared session
                    collector.session = self.session
                
                result = collector.run()
                
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing company {company['id']}: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"Error processing company {company['id']}: {str(e)}")
        
        # Step 3: Save the combined results
        combined_results = {
            "batch_collection": {
                "companies_processed": len(all_results),
                "companies_total": len(self.companies),
                "timestamp": datetime.now().isoformat()
            },
            "company_results": all_results
        }
        
        combined_file_path = os.path.join(self.base_dir, "batch_collection_results.json")
        with open(combined_file_path, "w") as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        logger.info(f"Batch document collection completed. Processed {len(all_results)}/{len(self.companies)} companies.")
        return combined_results


def main():
    # Use fixed config.json in the current directory
    config_path = "config.json"
    
    try:
        # Load the configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if this is a multi-company or single-company config
        if "companies" in config:
            # Multi-company configuration
            collector = BatchFinancialDocumentCollector(config_path)
            results = collector.run()
            
            # Print summary
            if results:
                print("\nBatch Collection Summary:")
                print(f"Companies processed: {results['batch_collection']['companies_processed']}/{results['batch_collection']['companies_total']}")
                
                for result in results['company_results']:
                    company = result['company']
                    print(f"\nCompany: {company['name']} ({company['id']})")
                    print(f"- Annual Reports: {len(result['annual_reports'])}")
                    print(f"- Concall Document Sets: {len(result['concalls'])}")
                
                print(f"\nAll documents saved to: {collector.base_dir}")
        else:
            # Single-company configuration (legacy mode)
            collector = FinancialDocumentCollector(config_path)
            results = collector.run()
            
            # Print summary
            if results:
                print("\nCollection Summary:")
                print(f"Company: {results['company']['name']} ({results['company']['id']})")
                print(f"- Annual Reports: {len(results['annual_reports'])}")
                print(f"- Concall Document Sets: {len(results['concalls'])}")
                
                if results['annual_reports']:
                    print("\nAnnual Reports:")
                    for report in results['annual_reports']:
                        print(f"- {report['year']} ({report['source']})")
                
                if results['concalls']:
                    print("\nConcall Documents:")
                    for concall in results['concalls']:
                        doc_types = [doc['type'] for doc in concall['documents']]
                        print(f"- {concall['period']}: {', '.join(doc_types)}")
                
                print(f"\nDocuments saved to: {collector.base_dir}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        print("Check the log file for details: financial_document_collection.log")


if __name__ == "__main__":
    main()