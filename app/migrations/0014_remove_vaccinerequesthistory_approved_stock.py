# Generated by Django 5.1 on 2024-08-10 09:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0013_vaccinerequest_dose_vaccinerequesthistory_dose'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='vaccinerequesthistory',
            name='approved_stock',
        ),
    ]
