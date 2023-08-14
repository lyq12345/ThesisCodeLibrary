import pymongo

client = pymongo.MongoClient(host='master', port=27017)
db = client['test']

collection = db.get_collection('testcollection')

student = {
    'id': '20170101',
    'name': 'Jordan',
    'age': 20,
    'gender': 'male'
}



result = collection.insert_one(student)
print(result)