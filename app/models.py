from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    USER_TYPE_CHOICES = [
        ('0', 'Parent'),
        ('1', 'Healthcare Provider'),
        ('admin', 'Admin'),
    ]
    
    usertype = models.CharField(
        max_length=20,
        choices=USER_TYPE_CHOICES,
        default='0'  # Default user type as 'Parent'
    )

    STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
    ]
    
    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default='active'  # Default status as 'Active'
    )

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_user_set',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        verbose_name='Groups',
    )
    
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_user_set',
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='User Permissions',
    )

class HealthProfile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    health_center_name = models.CharField(max_length=255)
    phone = models.CharField(max_length=20)
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100)
    license_number = models.CharField(max_length=50)

class ParentProfile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    contact_no = models.CharField(max_length=20)
    parentno = models.CharField(max_length=12, null=True, blank=True)  # New field added
    address = models.CharField(max_length=255)
    place = models.CharField(max_length=100)

            
class ChildProfile(models.Model):
    parent = models.ForeignKey(User, on_delete=models.CASCADE)
    child_name = models.CharField(max_length=255)
    dob = models.DateField()
    gender = models.CharField(max_length=10)
    blood_group = models.CharField(max_length=5)
    birth_weight = models.DecimalField(max_digits=5, decimal_places=2)
    birth_height = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)  # New field added
    current_weight = models.DecimalField(max_digits=5, decimal_places=2)
    current_height = models.DecimalField(max_digits=5, decimal_places=2)
    age = models.CharField(max_length=50, null=True, blank=True)

class VaccinationRecord(models.Model):
    child = models.ForeignKey(ChildProfile, on_delete=models.CASCADE)
    vaccine_taken = models.BooleanField(default=False)
    vaccine_name = models.CharField(max_length=255, blank=True, null=True)
    place = models.CharField(max_length=255, blank=True, null=True)
    date = models.DateField(blank=True, null=True)
    weight = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)  # New field added

class Vaccine(models.Model):
    vaccine_id = models.AutoField(primary_key=True)
    vaccine_name = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100)
    batch_number = models.CharField(max_length=6)
    date_manufacture = models.DateField()
    expiry_date = models.DateField()
    age_group = models.CharField(max_length=50)
    indications = models.TextField()
    stock = models.IntegerField()
    free_or_paid = models.CharField(max_length=4, choices=[('free', 'Free'), ('paid', 'Paid')])
    rate = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    def __str__(self):
        return self.vaccine_name

class VaccineDose(models.Model):
    dose_id = models.AutoField(primary_key=True)
    vaccine = models.ForeignKey(Vaccine, on_delete=models.CASCADE, related_name='vaccine_doses')  # Changed related_name
    dose_number = models.IntegerField()  # E.g., 1 for the first dose, 2 for the second dose, etc.
    date_administered = models.DateField(null=True, blank=True)  # Date when the dose is administered
    interval_days = models.CharField(max_length=50)  # Days between this dose and the previous dose
 

    def __str__(self):
        return f"{self.vaccine.vaccine_name} - Dose {self.dose_number}"

   
class VaccineRequest(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]
    
    healthcenter = models.ForeignKey('HealthProfile', on_delete=models.CASCADE)
    vaccine = models.ForeignKey(Vaccine, on_delete=models.CASCADE)
    dose = models.ForeignKey(VaccineDose, on_delete=models.CASCADE, null=True, blank=True)
    requested_stock = models.PositiveIntegerField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')
    request_date = models.DateTimeField(auto_now_add=True)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    approval_date = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.healthcenter} - {self.vaccine} - {self.dose} - {self.status}"


class VaccineRequestHistory(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]
    
    healthcenter = models.ForeignKey('HealthProfile', on_delete=models.CASCADE)
    vaccine = models.ForeignKey(Vaccine, on_delete=models.CASCADE)
    dose = models.ForeignKey(VaccineDose, on_delete=models.CASCADE, null=True, blank=True)
    requested_stock = models.PositiveIntegerField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')
    request_date = models.DateTimeField(auto_now_add=True)
    approval_date = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.healthcenter} - {self.vaccine} - {self.dose} - {self.status}"

class Appointment(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Confirmed', 'Confirmed'),
        ('Completed', 'Completed'),
        ('Cancelled', 'Cancelled'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # The person scheduling or attending the appointment
    health_center = models.ForeignKey(HealthProfile, on_delete=models.CASCADE)  # The health center for the appointment
    vaccine = models.ForeignKey(Vaccine, on_delete=models.CASCADE)  # The vaccine to be administered
    appointment_date = models.DateField()  # The date of the appointment
    appointment_time = models.TimeField()  # The time of the appointment
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')  # Status of the appointment
    approval_date = models.DateTimeField(null=True, blank=True)  # Date when the appointment is approved by the health center

    def __str__(self):
        return f"Appointment for {self.user} at {self.health_center} "

# Add this new model at the end of the file
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    related_appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Notification for {self.user.username}: {self.message[:50]}..."

class FeedingChart(models.Model):
    age = models.CharField(max_length=50, verbose_name="Age")  # Removed the choices parameter
    main_heading = models.CharField(max_length=255, verbose_name="Main Heading")
    description = models.TextField(verbose_name="Description")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.main_heading

# Model to store mental health details
class MentalHealthDetails(models.Model):
    age = models.CharField(max_length=100)
    image = models.ImageField(upload_to='mental_health_images/')

class MentalHealthDescription(models.Model):
    mental_health_detail = models.ForeignKey(MentalHealthDetails, related_name='descriptions', on_delete=models.CASCADE)
    description = models.TextField()


class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Product(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    product_name = models.CharField(max_length=150)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.product_name

class ProductDescription(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='descriptions')
    title = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return f"Description for {self.product.product_name}"

class ProductImage(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='product_images/')

    def __str__(self):
        return f"Image for {self.product.product_name}"

class CartItem(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey('Product', on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    date_added = models.DateTimeField(auto_now_add=True)

    def get_total_price(self):
        return self.product.price * self.quantity

    class Meta:
        ordering = ['-date_added']
        
class Wishlist(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    date_added = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'product')  # Prevents duplicate wishlist items