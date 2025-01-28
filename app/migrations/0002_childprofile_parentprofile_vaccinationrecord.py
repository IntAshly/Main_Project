# Generated by Django 5.0.6 on 2024-07-28 09:43

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChildProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('child_name', models.CharField(max_length=255)),
                ('dob', models.DateField()),
                ('gender', models.CharField(max_length=10)),
                ('blood_group', models.CharField(max_length=5)),
                ('birth_weight', models.DecimalField(decimal_places=2, max_digits=5)),
                ('current_weight', models.DecimalField(decimal_places=2, max_digits=5)),
                ('current_height', models.DecimalField(decimal_places=2, max_digits=5)),
                ('parent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='ParentProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contact_no', models.CharField(max_length=20)),
                ('address', models.CharField(max_length=255)),
                ('place', models.CharField(max_length=100)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='VaccinationRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vaccine_taken', models.BooleanField(default=False)),
                ('vaccine_name', models.CharField(blank=True, max_length=255, null=True)),
                ('place', models.CharField(blank=True, max_length=255, null=True)),
                ('date', models.DateField(blank=True, null=True)),
                ('child', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='app.childprofile')),
            ],
        ),
    ]
