from flask_login import UserMixin
from bson.objectid import ObjectId
from extensions import login_manager, users_collection, recruiters_collection


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']


class Recruiter(UserMixin):
    def __init__(self, recruiter_data):
        self.id = str(recruiter_data['_id'])
        self.email = recruiter_data['email']
        self.password_hash = recruiter_data['password_hash']
        self.company = recruiter_data.get('company', '')
        self.name = recruiter_data.get('name', '')


@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)

    recruiter_data = recruiters_collection.find_one({'_id': ObjectId(user_id)})
    if recruiter_data:
        return Recruiter(recruiter_data)

    return None
