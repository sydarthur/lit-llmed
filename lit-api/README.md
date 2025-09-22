# Literature Fetcher

A professional academic literature metadata collection and management system with direct Zotero integration.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test system works
python tests/test_connection.py --email your@email.com

# 3. Fetch articles from a journal (creates organized outputs automatically)
python main.py --email your@email.com fetch-single 0735-3766

# 4. Setup Zotero integration (optional)
python tests/setup_wizard.py

# 5. Fetch with direct Zotero import
python main.py --email your@email.com fetch-single 0735-3766 --max-articles 5
# Then import the generated RIS file: output/exports/ris/[filename].ris
```

## 📁 Project Structure

```
lit-api/
├── 📄 main.py                    # CLI interface
├── 📄 get_latest_literature.py   # Core fetching engine  
├── 📄 multi_journal_fetcher.py   # Multi-journal processing
├── 📄 journal_config.py          # Journal configuration
├── 📄 scheduler.py               # Automation system
├── 📁 config/                    # Configuration files
│   ├── journal_configs.json      # Journal settings
│   ├── zotero_config.json        # Zotero API credentials
│   └── scheduler_config.json     # Automation settings
├── 📁 output/                    # All outputs organized
│   ├── data/                     # JSON article data
│   ├── exports/csv/              # CSV exports
│   ├── exports/ris/              # RIS files for Zotero
│   └── logs/                     # System logs
├── 📁 utils/                     # Professional utilities
│   ├── zotero_client.py          # Zotero API client
│   ├── ris_exporter.py           # RIS file generation
│   └── config_manager.py         # Configuration management
└── 📁 tests/                     # Testing & setup tools
    ├── setup_wizard.py           # Interactive setup
    └── test_connection.py        # System testing
```

## ⚙️ Features

### 🔍 **Literature Fetching**
- **Multi-API Integration**: CrossRef, OpenAlex, Unpaywall
- **Rich Metadata**: Titles, authors, abstracts, DOIs, open access links
- **Date Filtering**: Fetch only recent articles
- **Rate Limiting**: Respects API guidelines
- **Parallel Processing**: Efficient multi-journal fetching

### 📚 **Zotero Integration**
- **Direct API Import**: Automatic import to your Zotero library
- **Organized Collections**: Custom collection names per journal
- **Complete Metadata**: Full bibliographic information preserved
- **RIS Export**: Alternative import method via files

### 🔧 **Professional Features**
- **Modular Architecture**: Clean, maintainable codebase
- **Configuration Management**: Centralized settings
- **Automated Scheduling**: Set-and-forget literature collection
- **Multiple Export Formats**: JSON, CSV, RIS
- **Comprehensive Logging**: Full operation tracking
- **Error Handling**: Robust failure recovery

## 📊 Supported Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **JSON** | Complete metadata | System processing, backup |
| **CSV** | Spreadsheet format | Data analysis, reporting |
| **RIS** | Bibliography format | Zotero import, citation managers |
| **Direct Zotero** | API integration | Seamless library management |

## 🏗️ Setup & Configuration

### 1. **Initial Setup**
```bash
# Interactive setup wizard
python tests/setup_wizard.py
```

The wizard configures:
- ✅ Zotero API credentials
- ✅ Journal preferences  
- ✅ Automation settings
- ✅ Output organization

### 2. **Manual Configuration**

#### **Zotero API Setup**
1. Visit https://www.zotero.org/settings/keys
2. Create new private key with:
   - ✅ Allow library access
   - ✅ Allow write access
3. Save credentials via setup wizard or manually:

```json
// config/zotero_config.json
{
  "api_key": "your_api_key_here",
  "user_id": "your_user_id_here",
  "library_type": "user"
}
```

#### **Journal Configuration**
```json
// config/journal_configs.json
{
  "journals": [
    {
      "name": "Journal of Business Logistics",
      "issn": "0735-3766",
      "publisher": "Wiley",
      "subject_area": "Supply Chain Management", 
      "zotero_collection": "Supply Chain Research",
      "active": true,
      "max_articles_per_fetch": 50,
      "days_back": 30
    }
  ]
}
```

## 🖥️ CLI Commands

### **Single Journal Fetching**
```bash
# Basic fetch
python main.py --email your@email.com fetch-single 0735-3766

# Advanced options
python main.py --email your@email.com fetch-single 0735-3766 \
  --max-articles 20 \
  --days-back 7 \
  --output custom_output.json \
  --create-ris
```

### **Multi-Journal Fetching**
```bash
# Fetch all configured journals
python main.py --email your@email.com fetch-all

# With Zotero integration
python main.py --email your@email.com fetch-all --zotero

# Custom exports
python main.py --email your@email.com fetch-all \
  --no-csv \
  --workers 3 \
  --rate-limit 2.0
```

### **Journal Management**
```bash
# List journals
python main.py --email your@email.com config --list

# Add journal
python main.py --email your@email.com config --add \
  "Nature Methods" "1548-7091" "Nature" "Methods" "Methodology Papers"

# Toggle active status
python main.py --email your@email.com config --toggle 1548-7091
```

### **Automation**
```bash
# Run scheduler once
python main.py --email your@email.com schedule --run-once

# Start continuous scheduler
python main.py --email your@email.com schedule --start
```

## 🔄 Automation & Scheduling

### **Setup Automated Fetching**
```json
// config/scheduler_config.json
{
  "fetch_interval_hours": 24,
  "export_csv": true,
  "export_ris": true, 
  "export_zotero": true,
  "retention_days": 30,
  "max_workers": 5
}
```

### **Run as Service**
```bash
# One-time fetch
python main.py --email your@email.com schedule --run-once

# Continuous monitoring
nohup python main.py --email your@email.com schedule --start &
```

## 🧪 Testing & Validation

### **Test System**
```bash
# Comprehensive test
python tests/test_connection.py \
  --email your@email.com \
  --api-key your_zotero_key \
  --user-id your_zotero_id

# Quick connection test
python tests/test_connection.py --email your@email.com
```

### **Validate Setup**
```bash
# Test single journal fetch
python main.py --email your@email.com fetch-single 0028-0836 --max-articles 3

# Test with Journal of Business Logistics (known to work well)
python main.py --email your@email.com fetch-single 0735-3766 --max-articles 3

# Check outputs (automatically organized)
ls output/data/          # JSON files
ls output/exports/ris/   # RIS files for Zotero import
ls output/exports/csv/   # CSV files (if generated)
```

## 📈 Output Organization

All outputs are automatically organized:

```
output/
├── data/                     # Raw JSON article data
│   ├── 0735_3766_20250921_184506.json    # Individual journal files
│   └── all_journals_20250921_143022.json # Combined multi-journal
├── exports/
│   ├── csv/                  # Spreadsheet exports
│   │   └── all_journals_20250921_143022.csv
│   └── ris/                  # Zotero-ready files (import these!)
│       ├── 0735_3766_20250921_184506.ris
│       └── all_journals_20250921_143022.ris
└── logs/                     # Operation logs (when enabled)
    └── literature_fetcher.log
```

**File Naming Convention**: `{journal_issn}_{timestamp}.{format}` or `all_journals_{timestamp}.{format}`

## 🔧 Advanced Usage

### **Custom API Integration**
```python
from utils.zotero_client import ZoteroClient
from utils.ris_exporter import RISExporter
from get_latest_literature import LiteratureFetcher

# Direct API usage
client = ZoteroClient(api_key, user_id)
collection_key = client.create_collection("My Research")
client.import_articles(articles, collection_key)
```

### **Custom Journal Sources**
```python
from journal_config import JournalConfig, JournalConfigManager

# Add custom journal
config_manager = JournalConfigManager()
journal = JournalConfig(
    name="Custom Journal",
    issn="1234-5678", 
    publisher="Publisher",
    subject_area="Research Area",
    zotero_collection="My Collection"
)
config_manager.add_journal(journal)
```

## 🛠️ Dependencies

```bash
pip install -r requirements.txt
```

**Core Requirements:**
- `requests` - API communication
- `schedule` - Automation
- Standard library: `json`, `csv`, `pathlib`, `datetime`, `logging`

## 🆘 Troubleshooting

### **Common Issues**

1. **Zotero Import**
   - **RIS Method (Recommended)**: Import the `.ris` files from `output/exports/ris/` via Zotero's File → Import
   - **Direct API**: Requires setup wizard configuration with proper API permissions
   - **Browser Extension**: Use Zotero Connector to import individual DOI links

2. **No Articles Found**
   - Normal for very recent time periods (try `--days-back 60`)
   - Some journals may have different publication schedules
   - Verify ISSN format: `0735-3766` (with hyphen)

3. **Access Issues**
   - **Use institutional email** (like `you@auburn.edu`) for better API access
   - **Open access links** work better with academic email addresses
   - Some publishers restrict API access

4. **Rate Limiting**
   - Increase `--rate-limit` delay (default: 1.0 seconds)
   - Reduce `--workers` count for multi-journal fetching
   - APIs have daily request limits

### **Debug Mode**
```bash
# Detailed logging
python main.py --log-level DEBUG --email your@email.com fetch-single 0735-3766

# Test individual components
python tests/test_connection.py --email your@email.com
```

## 🔄 Migration from Old Version

If upgrading from the original version:

```bash
# Backup old data
cp *.json backup/

# Run new setup
python tests/setup_wizard.py

# Import old journal configs (manual)
# Edit config/journal_configs.json with your previous settings
```

## 🎯 Best Practices

1. **Email Setup**: Use institutional email (e.g., `@auburn.edu`) for better API access and open access links
2. **Start Simple**: Begin with single journal fetching before setting up automation
3. **Zotero Workflow**: Use RIS file import (most reliable) over direct API for regular use
4. **Organization**: Let the system auto-organize outputs in `output/` directories
5. **Testing**: Run `python tests/test_connection.py` periodically to verify API access
6. **Backup**: Your configs are in `config/` - back these up regularly

## 💡 **Pro Tips**

- **Journal of Business Logistics (0735-3766)** works excellently for testing
- **RIS files** in `output/exports/ris/` are ready for direct Zotero import
- **Auburn email** provides better access to subscription content
- **Institutional VPN** may improve access to paywalled content
- **Start with 3-5 articles** for testing before large batch operations

## 📞 Support & Next Steps

### **Getting Help**
- **Setup Issues**: `python tests/setup_wizard.py`
- **Connection Problems**: `python tests/test_connection.py --email your@email.com`
- **Configuration**: Check/edit files in `config/` directory
- **Outputs**: All files auto-organized in `output/` subdirectories

### **Recommended Workflow**
1. **Start**: `python main.py --email your@auburn.edu fetch-single 0735-3766 --max-articles 5`
2. **Check Output**: Files created in `output/data/` and `output/exports/ris/`
3. **Import to Zotero**: File → Import → Select the `.ris` file
4. **Scale Up**: Add more journals via config commands or setup wizard
5. **Automate**: Set up scheduling for regular literature collection

### **Success Indicators** ✅
- JSON files appear in `output/data/`
- RIS files appear in `output/exports/ris/`
- Articles import cleanly into Zotero
- Abstracts and open access links included when available

---

## 🎉 **Ready to Revolutionize Your Literature Collection!**

**This system transforms hours of manual literature searching into automated, organized, and Zotero-ready workflows.**