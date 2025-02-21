# Generated by Django 5.0.6 on 2024-08-03 04:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_childprofile_parentprofile_vaccinationrecord'),
    ]

    operations = [
        migrations.AddField(
            model_name='childprofile',
            name='birth_height',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True),
        ),
        migrations.AddField(
            model_name='parentprofile',
            name='parentno',
            field=models.CharField(blank=True, max_length=12, null=True),
        ),
        migrations.AddField(
            model_name='vaccinationrecord',
            name='weight',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True),
        ),
    ]
