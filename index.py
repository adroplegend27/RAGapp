#!/usr/bin/env python3
"""
RAG System Step 3: Index for Instant Search & Timeline Building
This script creates a searchable index from processed emails and documents,
enabling queries about document contents and timeline events.

Dependencies:
- whoosh (pip install whoosh)
- sqlite3 (included in Python standard library)
"""

import os
import json
import argparse
import sqlite3
from datetime import datetime
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, DATETIME, STORED
from whoosh.qparser import QueryParser, MultifieldParser

# Create directories to store our data
os.makedirs("data/indexes", exist_ok=True)
os.makedirs("data/database", exist_ok=True)

# Define schema for Whoosh indexing
schema = Schema(
    id=ID(stored=True, unique=True),
    type=TEXT(stored=True),  # email or attachment
    email_id=ID(stored=True),
    date=DATETIME(stored=True),
    from_field=TEXT(stored=True),
    subject=TEXT(stored=True),
    body=TEXT(stored=True),
    filename=TEXT(stored=True),
    content=TEXT(stored=True)  # Main searchable content (either email body or attachment text)
)

# Create or open index
if not os.listdir("data/indexes"):
    index = create_in("data/indexes", schema)
else:
    index = open_dir("data/indexes")

# Setup SQLite for timeline tracking
conn = sqlite3.connect("data/database/timeline.db")
cursor = conn.cursor()

# Create tables for our timeline data
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    type TEXT,
    email_id TEXT,
    date TEXT,
    from_field TEXT,
    subject TEXT,
    filename TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    event_type TEXT,
    event_date TEXT,
    details TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
''')

# Create FTS virtual table for full-text search
cursor.execute('''
CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
    id, 
    content, 
    tokenize = 'porter unicode61'
)
''')

conn.commit()


def index_document(data, doc_type="email"):
    """Index an email or attachment document in both Whoosh and SQLite"""
    
    # Generate a unique ID if not present
    if doc_type == "email":
        doc_id = data.get("email_id", str(hash(data["subject"] + data["date"])))
        content = data["body"]
        filename = None
    else:  # attachment
        doc_id = data.get("file_id", str(hash(data["filename"])))
        content = data["extracted_text"]
        filename = data["filename"]
        email_id = data.get("email_id", None)
    
    # Index in Whoosh
    writer = index.writer()
    
    if doc_type == "email":
        writer.add_document(
            id=doc_id,
            type="email",
            email_id=doc_id,
            date=datetime.strptime(data["date"], "%Y-%m-%d"),
            from_field=data["from"],
            subject=data["subject"],
            body=data["body"],
            content=data["body"]
        )
    else:  # attachment
        writer.add_document(
            id=doc_id,
            type="attachment",
            email_id=email_id if email_id else "",
            filename=filename,
            content=content
        )
    
    writer.commit()
    
    # Store in SQLite for timeline tracking
    if doc_type == "email":
        cursor.execute('''
        INSERT OR REPLACE INTO documents (id, type, email_id, date, from_field, subject, filename)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, doc_type, doc_id, data["date"], data["from"], data["subject"], None))
        
        # Add to FTS table
        cursor.execute('''
        INSERT OR REPLACE INTO content_fts (id, content)
        VALUES (?, ?)
        ''', (doc_id, data["body"]))
        
        # Record event: email received
        cursor.execute('''
        INSERT INTO events (document_id, event_type, event_date, details)
        VALUES (?, ?, ?, ?)
        ''', (doc_id, "received", data["date"], f"Email from {data['from']} with subject '{data['subject']}'"))
        
    else:  # attachment
        cursor.execute('''
        INSERT OR REPLACE INTO documents (id, type, email_id, date, from_field, subject, filename)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, doc_type, email_id if email_id else "", data.get("date", ""), "", "", filename))
        
        # Add to FTS table
        cursor.execute('''
        INSERT OR REPLACE INTO content_fts (id, content)
        VALUES (?, ?)
        ''', (doc_id, content))
        
        # Record event: attachment processed
        cursor.execute('''
        INSERT INTO events (document_id, event_type, event_date, details)
        VALUES (?, ?, ?, ?)
        ''', (doc_id, "processed", datetime.now().strftime("%Y-%m-%d"), f"Processed attachment {filename}"))
    
    conn.commit()


def process_data_directory(directory):
    """Process all JSON files in a directory and add them to the index"""
    files_processed = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                # Try UTF-8 first (most common for JSON)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                try:
                    # Try with utf-8-sig (for files with BOM)
                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                except UnicodeDecodeError:
                    try:
                        # Last resort: use latin-1 which can read any file
                        with open(file_path, "r", encoding="latin-1") as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {filename}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                
            try:
                # Determine if this is an email or attachment
                if "email_id" in data and "body" in data:
                    index_document(data, doc_type="email")
                    files_processed += 1
                    
                    # If this email has attachments, we assume they're also processed
                    if "attachments" in data and data["attachments"]:
                        for attachment in data["attachments"]:
                            # Find corresponding attachment file
                            attachment_name = os.path.splitext(attachment)[0]
                            potential_files = [f for f in os.listdir(directory) 
                                            if f.startswith(attachment_name) and f.endswith(".json")]
                            
                            for attachment_file in potential_files:
                                try:
                                    with open(os.path.join(directory, attachment_file), "r", encoding="utf-8") as af:
                                        attachment_data = json.load(af)
                                except UnicodeDecodeError:
                                    try:
                                        with open(os.path.join(directory, attachment_file), "r", encoding="latin-1") as af:
                                            attachment_data = json.load(af)
                                    except Exception as e:
                                        print(f"Error processing attachment {attachment_file}: {str(e)}")
                                        continue
                                
                                attachment_data["email_id"] = data["email_id"]
                                index_document(attachment_data, doc_type="attachment")
                                files_processed += 1
                                    
                elif "file_id" in data and "extracted_text" in data:
                    index_document(data, doc_type="attachment")
                    files_processed += 1
                else:
                    print(f"Skipping {filename}: Unrecognized format")
            except Exception as e:
                print(f"Error indexing {filename}: {str(e)}")
    
    return files_processed



def search_documents(query_string, limit=20):
    """Search documents using Whoosh"""
    with index.searcher() as searcher:
        query_parser = MultifieldParser(["subject", "body", "content", "from_field"], schema=index.schema)
        query = query_parser.parse(query_string)
        results = searcher.search(query, limit=limit)
        
        return [dict(result) for result in results]


def search_timeline(query_string, limit=20):
    """Search documents using SQLite FTS and return with timeline info"""
    cursor.execute('''
    SELECT c.id, c.content, d.type, d.date, d.from_field, d.subject, d.filename
    FROM content_fts c
    JOIN documents d ON c.id = d.id
    WHERE content_fts MATCH ?
    LIMIT ?
    ''', (query_string, limit))
    
    results = cursor.fetchall()
    timeline_data = []
    
    for row in results:
        doc_id, content, doc_type, date, from_field, subject, filename = row
        
        # Get timeline events for this document
        cursor.execute('''
        SELECT event_type, event_date, details
        FROM events
        WHERE document_id = ?
        ORDER BY event_date
        ''', (doc_id,))
        
        events = [{"type": e[0], "date": e[1], "details": e[2]} for e in cursor.fetchall()]
        
        timeline_data.append({
            "id": doc_id,
            "type": doc_type,
            "date": date,
            "from": from_field,
            "subject": subject,
            "filename": filename,
            "content_snippet": content[:200] + "..." if len(content) > 200 else content,
            "timeline": events
        })
    
    return timeline_data


def find_contradictions(topic, threshold=0.5):
    """Find potential contradictions on a specific topic"""
    # First get all documents related to the topic
    results = search_documents(topic, limit=100)
    
    # Group by sender/author
    statements_by_source = {}
    for result in results:
        source = result.get("from_field", "Unknown")
        if source not in statements_by_source:
            statements_by_source[source] = []
        
        statements_by_source[source].append({
            "date": result.get("date"),
            "content": result.get("content", ""),
            "document_id": result.get("id")
        })
    
    # Simple contradiction detection (can be improved with NLP tools)
    contradictions = []
    
    # For this basic implementation, we'll just look for specific keywords
    # that might indicate contradictions within the same source
    contradiction_terms = ["not", "never", "contrary", "oppose", "different", 
                          "changed", "incorrect", "wrong", "mistaken"]
    
    for source, statements in statements_by_source.items():
        # Sort by date
        statements.sort(key=lambda x: x["date"] if x["date"] else datetime.min)
        
        for i in range(len(statements) - 1):
            for j in range(i + 1, len(statements)):
                # Check for contradiction terms
                has_contradiction = False
                for term in contradiction_terms:
                    if term in statements[j]["content"].lower():
                        has_contradiction = True
                        break
                
                if has_contradiction:
                    contradictions.append({
                        "source": source,
                        "statement1": {
                            "date": statements[i]["date"],
                            "content": statements[i]["content"][:150] + "...",
                            "document_id": statements[i]["document_id"]
                        },
                        "statement2": {
                            "date": statements[j]["date"],
                            "content": statements[j]["content"][:150] + "...",
                            "document_id": statements[j]["document_id"]
                        }
                    })
    
    return contradictions


def display_search_results(results, query_type="basic"):
    """Format and display search results in a readable way"""
    if not results:
        print("No results found.")
        return
    
    if query_type == "basic":
        print(f"\nFound {len(results)} documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('subject', result.get('filename', 'Unnamed document'))}")
            print(f"   Type: {'Email' if result.get('type') == 'email' else 'Attachment'}")
            if result.get('date'):
                print(f"   Date: {result.get('date')}")
            if result.get('from_field'):
                print(f"   From: {result.get('from_field')}")
            print(f"   Content: {result.get('content', '')[:150]}...")
    
    elif query_type == "timeline":
        print(f"\nTimeline for {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc['subject'] or doc['filename']} ({doc['date']})")
            print(f"   From: {doc['from']}")
            print(f"   Content: {doc['content_snippet']}")
            print("   Timeline events:")
            for event in doc['timeline']:
                print(f"   - {event['date']}: {event['type']} - {event['details']}")
    
    elif query_type == "contradictions":
        print(f"\nFound {len(results)} potential contradictions:")
        for i, contradiction in enumerate(results, 1):
            print(f"\n{i}. From: {contradiction['source']}")
            print(f"   Initial statement ({contradiction['statement1']['date']}): {contradiction['statement1']['content']}")
            print(f"   Later statement ({contradiction['statement2']['date']}): {contradiction['statement2']['content']}")


def close_connections():
    """Close database connections when done"""
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='RAG System: Index and search documents')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents from a directory')
    index_parser.add_argument('directory', help='Directory containing JSON files to index')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
    
    # Timeline command
    timeline_parser = subparsers.add_parser('timeline', help='Search and show document timeline')
    timeline_parser.add_argument('query', help='Search query')
    timeline_parser.add_argument('--limit', type=int, default=10, help='Maximum number of results')
    
    # Contradictions command
    contradictions_parser = subparsers.add_parser('contradictions', help='Find contradictions on a topic')
    contradictions_parser.add_argument('topic', help='Topic to analyze for contradictions')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        files_processed = process_data_directory(args.directory)
        print(f"Successfully indexed {files_processed} files from {args.directory}")
    
    elif args.command == 'search':
        results = search_documents(args.query, args.limit)
        display_search_results(results, "basic")
    
    elif args.command == 'timeline':
        results = search_timeline(args.query, args.limit)
        display_search_results(results, "timeline")
    
    elif args.command == 'contradictions':
        results = find_contradictions(args.topic)
        display_search_results(results, "contradictions")
    
    else:
        parser.print_help()
    
    close_connections()


if __name__ == "__main__":
    main()
