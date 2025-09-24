import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, jsonify, request
import logging

# Add the project root to Python path for imports
sys.path.append('/app')

from src.core.models import Journal, Article
from src.gcp.gcs_storage import GCSStore
from src.gcp.gcp_fetch import GCPCrossrefClient

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'lit-llmed-gcp',
        'version': '1.0.0'
    })

@app.route('/api/status')
def status():
    """Service status endpoint"""
    return jsonify({
        'service': 'lit-llmed-gcp',
        'status': 'running',
        'environment': os.getenv('ENV', 'development')
    })

@app.route('/api/journals')
def list_journals():
    """List configured journals"""
    try:
        # Load journals from the embedded config
        journals_config = {
            "journals": [
                {
                    "name": "Journal of Business Logistics",
                    "issn": "0735-3766",
                    "publisher": "Wiley",
                    "subject_area": "Supply Chain Management",
                    "zotero_collection": "Automation",
                    "fetch_abstracts": True,
                    "fetch_oa_links": True,
                    "max_articles_per_fetch": 50,
                    "days_back": 30,
                    "active": True
                },
                {
                    "name": "Production and Operations Management",
                    "issn": "1059-1478",
                    "publisher": "Sage",
                    "subject_area": "Operations Management",
                    "zotero_collection": "Automation",
                    "fetch_abstracts": True,
                    "fetch_oa_links": True,
                    "max_articles_per_fetch": 50,
                    "days_back": 30,
                    "active": True
                },
                {
                    "name": "Journal of Operations Management",
                    "issn": "0272-6963",
                    "publisher": "Wiley",
                    "subject_area": "Operations Management",
                    "zotero_collection": "Automation",
                    "fetch_abstracts": True,
                    "fetch_oa_links": True,
                    "max_articles_per_fetch": 50,
                    "days_back": 30,
                    "active": True
                }
            ]
        }
        
        return jsonify({
            'journals': journals_config['journals'],
            'count': len(journals_config['journals'])
        })
        
    except Exception as e:
        logger.error(f"Error listing journals: {str(e)}")
        return jsonify({
            'error': 'Failed to list journals',
            'message': str(e)
        }), 500

@app.route('/api/fetch', methods=['POST'])
def fetch_literature():
    """Fetch literature for all configured journals"""
    try:
        # Get email from environment or request
        email = os.getenv('FETCHER_EMAIL', 'lit-llmed@example.com')
        
        # Get optional parameters
        data = request.get_json() or {}
        days_back = data.get('days_back', 30)
        max_articles = data.get('max_articles_per_fetch', 50)
        
        logger.info(f"Starting literature fetch - email: {email}, days_back: {days_back}")
        
        # Create journals from config
        journals = [
            Journal(
                name="Journal of Business Logistics",
                issn="0735-3766",
                publisher="Wiley",
                subject_area="Supply Chain Management",
                zotero_collection="Automation",
                fetch_abstracts=True,
                fetch_oa_links=True,
                max_articles_per_fetch=max_articles,
                days_back=days_back,
                active=True
            ),
            Journal(
                name="Production and Operations Management",
                issn="1059-1478",
                publisher="Sage",
                subject_area="Operations Management",
                zotero_collection="Automation",
                fetch_abstracts=True,
                fetch_oa_links=True,
                max_articles_per_fetch=max_articles,
                days_back=days_back,
                active=True
            ),
            Journal(
                name="Journal of Operations Management",
                issn="0272-6963",
                publisher="Wiley",
                subject_area="Operations Management",
                zotero_collection="Automation",
                fetch_abstracts=True,
                fetch_oa_links=True,
                max_articles_per_fetch=max_articles,
                days_back=days_back,
                active=True
            )
        ]
        
        # Initialize GCP fetch client
        fetch_client = GCPCrossrefClient(email, rate_limit_delay=1.0)
        
        # Fetch articles for each journal
        results = {}
        for journal in journals:
            articles = fetch_client.fetch_latest(journal)
            results[journal.issn] = articles
        
        # Initialize GCS storage
        gcs_store = GCSStore()
        gcs_store.ensure_bucket_exists()
        
        # Generate timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Upload results to GCS
        uploaded_files = {}
        total_articles = 0
        
        for issn, articles in results.items():
            if articles:
                journal = next(j for j in journals if j.issn == issn)
                filename = f"{journal.name.lower().replace(' ', '_')}_{timestamp}.json"
                gs_path = gcs_store.upload_articles(articles, filename)
                uploaded_files[journal.name] = {
                    'gcs_path': gs_path,
                    'article_count': len(articles)
                }
                total_articles += len(articles)
        
        # Upload combined results
        all_articles = [article for articles in results.values() for article in articles]
        if all_articles:
            combined_filename = f"all_journals_{timestamp}.json"
            combined_gs_path = gcs_store.upload_articles(all_articles, combined_filename)
            uploaded_files['combined'] = {
                'gcs_path': combined_gs_path,
                'article_count': len(all_articles)
            }
        
        # Create and upload summary
        summary = {
            'fetch_timestamp': timestamp,
            'total_articles': total_articles,
            'journals_fetched': len([r for r in results.values() if r]),
            'uploaded_files': uploaded_files,
            'fetch_parameters': {
                'days_back': days_back,
                'max_articles_per_fetch': max_articles,
                'email': email
            }
        }
        
        summary_gs_path = gcs_store.upload_summary(summary, timestamp)
        
        logger.info(f"Literature fetch completed - total articles: {total_articles}")
        
        return jsonify({
            'status': 'success',
            'message': f'Fetched {total_articles} articles from {len(journals)} journals',
            'summary': summary,
            'summary_gcs_path': summary_gs_path
        })
        
    except Exception as e:
        logger.error(f"Error fetching literature: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch literature',
            'message': str(e)
        }), 500

@app.route('/api/fetch/<issn>', methods=['POST'])
def fetch_single_journal(issn: str):
    """Fetch literature for a single journal by ISSN"""
    try:
        email = os.getenv('FETCHER_EMAIL', 'lit-llmed@example.com')
        
        # Find journal by ISSN
        journal_configs = {
            "0735-3766": {
                "name": "Journal of Business Logistics",
                "publisher": "Wiley",
                "subject_area": "Supply Chain Management"
            },
            "1059-1478": {
                "name": "Production and Operations Management", 
                "publisher": "Sage",
                "subject_area": "Operations Management"
            },
            "0272-6963": {
                "name": "Journal of Operations Management",
                "publisher": "Wiley",
                "subject_area": "Operations Management"
            }
        }
        
        if issn not in journal_configs:
            return jsonify({
                'error': 'Journal not found',
                'message': f'ISSN {issn} not in configured journals'
            }), 404
        
        config = journal_configs[issn]
        data = request.get_json() or {}
        
        journal = Journal(
            name=config["name"],
            issn=issn,
            publisher=config["publisher"],
            subject_area=config["subject_area"],
            zotero_collection="Automation",
            fetch_abstracts=True,
            fetch_oa_links=True,
            max_articles_per_fetch=data.get('max_articles_per_fetch', 50),
            days_back=data.get('days_back', 30),
            active=True
        )
        
        # Fetch articles
        fetch_client = GCPCrossrefClient(email, rate_limit_delay=1.0)
        articles = fetch_client.fetch_latest(journal)
        
        # Upload to GCS
        if articles:
            gcs_store = GCSStore()
            gcs_store.ensure_bucket_exists()
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{journal.name.lower().replace(' ', '_')}_{timestamp}.json"
            gs_path = gcs_store.upload_articles(articles, filename)
            
            return jsonify({
                'status': 'success',
                'journal': journal.name,
                'issn': issn,
                'article_count': len(articles),
                'gcs_path': gs_path,
                'articles': [article.model_dump() for article in articles[:5]]  # Show first 5
            })
        else:
            return jsonify({
                'status': 'success',
                'journal': journal.name,
                'issn': issn,
                'article_count': 0,
                'message': 'No new articles found'
            })
            
    except Exception as e:
        logger.error(f"Error fetching single journal {issn}: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch journal',
            'message': str(e)
        }), 500

@app.route('/api/storage/list')
def list_storage_files():
    """List files in GCS storage"""
    try:
        folder = request.args.get('folder', 'data')
        prefix = request.args.get('prefix', '')
        
        gcs_store = GCSStore()
        files = gcs_store.list_files(folder, prefix)
        
        return jsonify({
            'folder': folder,
            'prefix': prefix,
            'files': files,
            'count': len(files)
        })
        
    except Exception as e:
        logger.error(f"Error listing storage files: {str(e)}")
        return jsonify({
            'error': 'Failed to list storage files',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)