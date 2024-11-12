from django.contrib.auth.models import AbstractUser
from django.db import models


# Create your models here.
class CustomUser(AbstractUser):
    steam_id = models.CharField(max_length=50, blank=True, null=True, unique=True)

    def __str__(self):
        return self.username
