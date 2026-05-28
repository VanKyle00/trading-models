"""Data loaders, one subpackage per asset class. Each loader pulls from its
source, canonicalizes the schema, and caches parquet under
``data/processed/<source>/``."""
