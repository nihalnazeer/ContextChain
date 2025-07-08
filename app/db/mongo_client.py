from pymongo import MongoClient

def get_mongo_client(uri):
    return MongoClient(uri)