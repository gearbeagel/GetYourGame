from allauth.socialaccount.models import SocialAccount
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.contrib.auth import get_user_model


@receiver(post_save, sender=SocialAccount)
def update_user_steam_data(sender, instance, **kwargs):
    if instance.provider == 'steam':
        user = instance.user
        steam_id = instance.extra_data['steamid']
        user.steam_id = steam_id
        user.save()
