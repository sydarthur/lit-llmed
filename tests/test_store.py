from datetime import datetime

from src.core.models import Article, Author, OpenAccess
from src.core.store import ContentStore


def make_article(title: str) -> Article:
    return Article(
        title=title,
        doi="10.1234/abc",
        journal="Journal",
        issn="1234-5678",
        published=datetime(2023, 1, 1),
        authors=[Author(name="Jane Doe")],
        open_access=OpenAccess(url="https://example.com"),
    )


def test_write_and_read_json(tmp_path):
    store = ContentStore(output_root=tmp_path / "output")
    articles = [make_article("A"), make_article("B")]
    path = store.write_json(articles, filename="sample.json", use_content_hash=False)
    loaded = store.read_json(path)
    assert len(loaded) == 2
    assert loaded[0].title == "A"


def test_export_csv(tmp_path):
    store = ContentStore(output_root=tmp_path / "output")
    articles = [make_article("A")]
    csv_path = store.export_csv(articles, path=tmp_path / "export.csv")
    assert csv_path.exists()
    assert csv_path.read_text()
