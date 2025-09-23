from datetime import datetime

from src.core.models import Article, Author, Journal, OpenAccess


def test_author_from_parts():
    author = Author.from_parts("Jane", "Doe")
    assert author.name == "Jane Doe"
    assert author.given == "Jane"
    assert author.family == "Doe"


def test_article_serialization_roundtrip(tmp_path):
    article = Article(
        title="Sample",
        doi="10.1234/abc",
        journal="Journal",
        issn="1234-5678",
        published=datetime(2023, 1, 1),
        authors=[Author(name="Jane Doe")],
        open_access=OpenAccess(url="https://example.com"),
    )
    data = article.model_dump()
    clone = Article.model_validate(data)
    assert clone.title == article.title
    assert clone.open_access.url == article.open_access.url


def test_journal_defaults():
    journal = Journal(name="Test", issn="0000-0000")
    assert journal.fetch_abstracts
    assert journal.max_articles_per_fetch == 50
