# from neomodel import StructuredNode, StringProperty, RelationshipTo

# class Person(StructuredNode):
#     name = StringProperty(unique_index=True)
#     friends = RelationshipTo('Person', 'FRIEND')

# # Usage
# person1 = Person(name='Alice').save()
# person2 = Person(name='Bob').save()

# person1.friends.connect(person2)

# # Querying
# alice_friends = person1.friends.all()
# myapp/models.py
from django.db import models
from django.contrib.auth.models import User

class Question(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question_text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.question_text[:50]}"

    class Meta:
        ordering = ['-timestamp']




