import imaplib
import email
import os
import json
import base64
from datetime import datetime
from email.header import decode_header

def connect_to_mail_server(username, password, server="imap.gmail.com"):
    """
    Connect to email server using IMAP
    """
    mail = imaplib.IMAP4_SSL(server)
    mail.login(username, password)
    return mail

def get_emails(mail, folder="INBOX", limit=None, criteria="FROM "):
    """
    Fetch emails from specified folder with optional limit
    """
    mail.select(folder)
    status, messages = mail.search(None, criteria)
    email_ids = messages[0].split()
    
    # If limit is specified, only get the most recent emails
    if limit and len(email_ids) > limit:
        email_ids = email_ids[-limit:]
        
    return email_ids

def save_attachment(part, email_id, attachment_dir):
    """
    Save email attachment to disk and return filename
    """
    filename = part.get_filename()
    
    if filename:
        # Clean up filename if needed
        filename = decode_filename(filename)
        
        # Create directory if it doesn't exist
        if not os.path.isdir(attachment_dir):
            os.makedirs(attachment_dir)
            
        filepath = os.path.join(attachment_dir, f"{email_id}_{filename}")
        
        # Get attachment data and save
        payload = part.get_payload(decode=True)
        if payload:
            with open(filepath, 'wb') as f:
                f.write(payload)
            return filename
    
    return None

def decode_filename(filename):
    """
    Decode filename from email encoding
    """
    if isinstance(filename, str):
        return filename
        
    decoded_parts = decode_header(filename)
    filename_parts = []
    
    for content, encoding in decoded_parts:
        if isinstance(content, bytes):
            if encoding:
                filename_parts.append(content.decode(encoding or 'utf-8', errors='replace'))
            else:
                filename_parts.append(content.decode('utf-8', errors='replace'))
        else:
            filename_parts.append(content)
            
    return ''.join(filename_parts)

def parse_date(date_str):
    """
    Parse email date to consistent format
    """
    if not date_str:
        return None
        
    try:
        # Try to parse with email.utils
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        # Fallback to simple format
        try:
            return datetime.strptime(date_str[:16], '%a, %d %b %Y').strftime('%Y-%m-%d')
        except:
            return date_str

def process_email(mail, email_id, attachment_dir="attachments"):
    """
    Process a single email and return as JSON object
    """
    status, data = mail.fetch(email_id, "(RFC822)")
    raw_email = data[0][1]
    
    # Parse the raw email
    msg = email.message_from_bytes(raw_email)
    
    # Extract basic info
    email_from = msg.get("From", "")
    subject = decode_filename(msg.get("Subject", ""))
    date = parse_date(msg.get("Date"))
    
    # Extract body and attachments
    body = ""
    attachments = []
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Extract text body
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    body_bytes = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    body = body_bytes.decode(charset, errors='replace')
                except:
                    body = "Error decoding message body"
            
            # Handle attachments
            elif "attachment" in content_disposition or "filename" in content_disposition:
                filename = save_attachment(part, email_id.decode(), attachment_dir)
                if filename:
                    attachments.append(filename)
    else:
        # Handle plain text emails
        try:
            body_bytes = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or 'utf-8'
            body = body_bytes.decode(charset, errors='replace')
        except:
            body = "Error decoding message body"
    
    # Create JSON object
    email_json = {
        "email_id": email_id.decode(),
        "date": date,
        "from": email_from,
        "subject": subject,
        "body": body,
        "attachments": attachments
    }
    
    return email_json

def main():
    # Configuration
    username = ""  # Replace with your email
    password = ""  # Use app password for Gmail
    server = "imap.gmail.com"  # Replace with your email server
    folder = "INBOX"
    output_dir = "emails_data"
    attachment_dir = "attachments"
    limit = 10  # Set to None to fetch all emails
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Connect to mail server
    mail = connect_to_mail_server(username, password, server)
    
    # Get emails
    email_ids = get_emails(mail, folder, limit)
    print(f"Found {len(email_ids)} emails to process")
    
    # Process each email
    for i, email_id in enumerate(email_ids):
        try:
            print(f"Processing email {i+1}/{len(email_ids)} (ID: {email_id.decode()})")
            email_json = process_email(mail, email_id, attachment_dir)
            
            # Save JSON to file
            json_filename = os.path.join(output_dir, f"email_{email_id.decode()}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(email_json, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error processing email {email_id.decode()}: {str(e)}")
    
    # Logout
    mail.logout()
    print("Email processing complete")

if __name__ == "__main__":
    main()