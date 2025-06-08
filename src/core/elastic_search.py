from elasticsearch import Elasticsearch
import os


def get_elasticsearch_client() -> Elasticsearch:
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    es_client = Elasticsearch([es_url])
    if not es_client.ping():
        raise ConnectionError(f"Failed to connect to Elasticsearch at {es_url}")
    print(f"Connected to Elasticsearch at {es_url}")
    return es_client