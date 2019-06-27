from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class TrainAppName(db.Model):
    __tablename__ = 'train_app_name'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, default=0)


    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name
        }

class InferenceAppName(db.Model):
    __tablename__ = 'inference_app_name'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, default=0)


    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name
        }