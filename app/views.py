from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.hashers import make_password
from .models import User, HealthProfile
from django.contrib import messages
from django.core.mail import send_mail
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required
from .models import User, ParentProfile, ChildProfile, VaccinationRecord
from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from .models import Vaccine, VaccineDose, VaccineRequest, VaccineRequestHistory, HealthProfile
from django.contrib.auth.decorators import login_required
from datetime import datetime
from django.utils import timezone
from django.http import HttpResponseForbidden
from django.urls import reverse
from .models import Appointment
from django.http import HttpResponse
import logging
from django.utils.dateparse import parse_date, parse_time
from .models import Appointment
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth import get_user_model
from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_str 
from .models import Notification
from datetime import date
from django.contrib.auth.signals import user_logged_out
from django.dispatch import receiver
from django.contrib.auth.signals import user_logged_out
from django.dispatch import receiver
from django.http import JsonResponse
from app.ml_models import predict_vaccine_details  # Import the ML function
from django.conf import settings
import os
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .models import FeedingChart
from .models import MentalHealthDetails
from django.core.files.storage import FileSystemStorage
from .models import MentalHealthDetails, MentalHealthDescription
from django.core.exceptions import ObjectDoesNotExist
from dateutil.relativedelta import relativedelta
from .models import Category, Product, ProductDescription, ProductImage
from django.db import transaction
from django.http import JsonResponse
from .models import CartItem, Product
from django.views.decorators.http import require_POST
from .models import Wishlist

def register_view(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        password = request.POST['password']

        if User.objects.filter(username=email).exists():
            messages.error(request, 'An account with this email already exists.')
            return redirect('register')

        # Create a user
        user = User.objects.create_user(
            username=email, 
            first_name=first_name, 
            last_name=last_name, 
            email=email,  
            password=password
        )
        user.save()

        # Authenticate and log in the user
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Account created for {email}!')
            return redirect('role')  # Redirect to the role selection page
        
    return render(request, 'register.html')

def role_view(request): 
    if request.method == 'POST':
        role = request.POST.get('role')  # Use get to avoid KeyError if role is missing
        if role:  # Ensure the role is provided
            if request.user.is_authenticated:
                user = request.user  # The authenticated user is already accessible via request.user
                
                # Handling roles
                if role == 'parent':
                    user.usertype = 'parent'
                    user.status = 'active'
                    user.save()
                    return redirect('login')  # Redirect to login page for parents

                elif role == 'healthcare_provider':
                    user.usertype = 'healthcare_provider'
                    user.status = 'inactive'
                    user.save()
                    return redirect('health_profile_cmplt')  # Redirect to health profile page

            else:
                messages.error(request, 'User is not authenticated.')
                return redirect('login')  # Redirect to login page if not authenticated
        
        else:
            messages.error(request, 'Role not provided.')
            return redirect('role')  # Redirect to role selection if no role is provided
    
    return render(request, 'role.html')

def is_profile_completed(user):
    # Check if the user has a completed child profile
    return ChildProfile.objects.filter(parent=user).exists()

def login_view(request):
    if request.method == 'POST':
        username = request.POST['email']
        password = request.POST['password']

        # Check for admin credentials first
        if username == 'nurturenest@gmail.com' and password == 'Admin@123':
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('admin_home')  # Redirect to admin home page

        user = authenticate(request, username=username, password=password)
        if user is not None:
            if user.status == 'inactive':
                return render(request, 'login.html', {'error_message': 'Your account is inactive. Please wait for approval.'})
            login(request, user)
            if user.usertype == 'parent':  # Parent
                if is_profile_completed(user):
                    return redirect('home')  # Redirect to home page if profile is complete
                else:
                    return redirect('child_profile')  # Redirect to child profile completion page
            elif user.usertype == 'healthcare_provider':  # Healthcare provider
                return redirect('health_home')  # Redirect to healthcare provider home page
        else:
            return render(request, 'login.html', {'error_message': 'Invalid Credentials!'})
    return render(request, 'login.html')

def about(request):
    return render(request, 'about.html')

def child_profile_view(request):
    if request.method == 'POST':
        contact_no = request.POST.get('contact_no')
        parentno = request.POST.get('parentno')  # New field added
        address = request.POST.get('address')
        place = request.POST.get('place')

        child_name = request.POST.get('child_name')
        dob = request.POST.get('dob')
        gender = request.POST.get('gender')
        blood_group = request.POST.get('blood_group')
        birth_weight = request.POST.get('birth_weight')
        birth_height = request.POST.get('birth_height')
        age = request.POST.get('age')
        current_weight = request.POST.get('current_weight')
        current_height = request.POST.get('current_height')

        previous_vaccinations = request.POST.get('previous_vaccinations')
        vaccine_name = request.POST.get('vaccine_name')
        vaccine_date = request.POST.get('vaccine_date')
        vaccine_place = request.POST.get('vaccine_place')
        weight = request.POST.get('weight')  # New field added

        # Ensure the user is authenticated
        if request.user.is_authenticated:
            # Save Parent Details
            parent_details = ParentProfile.objects.create(
                user=request.user,
                contact_no=contact_no,
                parentno=parentno,  # New field added
                address=address,
                place=place
            )

            # Save Child Profile
            child_profile = ChildProfile.objects.create(
                parent=request.user,
                child_name=child_name,
                dob=dob,
                gender=gender,
                blood_group=blood_group,
                birth_weight=birth_weight,
                birth_height=birth_height,
                age=age,
                current_weight=current_weight,
                current_height=current_height,
            )

            # Save Vaccination Records if applicable
            if previous_vaccinations == 'yes' and vaccine_name and vaccine_date and vaccine_place:
                vaccination_record = VaccinationRecord.objects.create(
                    child=child_profile,
                    vaccine_taken=True,
                    vaccine_name=vaccine_name,
                    date=vaccine_date,
                    place=vaccine_place,
                    weight=weight  # New field added
                )

            messages.success(request, 'Child profile and vaccination details saved successfully.')
            return redirect('home')
        else:
            messages.error(request, 'User not authenticated.')
            return redirect('login')

    context = {
        'parent_name': request.user.get_full_name(),
        'parent_email': request.user.email
    }
    return render(request, 'child_profile.html', context)

def health_profile_cmplt(request):
    if request.method == 'POST':
        health_center_name = request.POST.get('health_center_name')  # Correctly accessing form fields
        # email = request.POST.get('email')
        phone = request.POST.get('phone')
        address = request.POST.get('address')
        city = request.POST.get('city')
        license_number = request.POST.get('license')

        HealthProfile.objects.create(
            user=request.user,
            health_center_name=health_center_name,
            # email=email,
            phone=phone,
            address=address,
            city=city,
            license_number=license_number
        )

        messages.success(request, 'Health profile saved.')
        return redirect('index')

    return render(request, 'health_profile_cmplt.html')

def request_view(request):
    health_profiles = HealthProfile.objects.filter(user__status='inactive')  # Ensure fetching inactive health profiles
    context = {'health_profiles': health_profiles}
    return render(request, 'request.html', context)  # Ensure the context is passed correctly

def approve_health_center(request, pk):
    if request.method == 'POST':
        health_profile = get_object_or_404(HealthProfile, pk=pk)
        user = health_profile.user
        user.status = 'active'
        user.save()
        
        # Send approval email
        send_mail(
            'NurtureNest Health Center Approval',
            'Your health center has been approved. You can now log in.',
            'nurturenest02@example.com',
            [user.email],
            fail_silently=False,
        )
        
        messages.success(request, f'Health center {health_profile.health_center_name} approved.')
        return redirect('request')

def reject_health_center(request, pk):
    if request.method == 'POST':
        health_profile = get_object_or_404(HealthProfile, pk=pk)
        user = health_profile.user
        
        # Send rejection email
        send_mail(
            'NurtureNest Health Center Rejection',
            'Your health center registration has been rejected.',
            'nurturenest02@example.com',
            [user.email],
            fail_silently=False,
        )
        
        # Optionally delete the profile and user
        health_profile.delete()
        user.delete()
        
        messages.success(request, f'Health center {health_profile.health_center_name} rejected.')
        return redirect('request')



def user_index(request):
    return render(request, 'user/userindex.html')

def logout_view(request):
    logout(request)
    return redirect('index')

def index_view(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')

def admin_view(request):
    return render(request, 'admin.html')

def admin_home_view(request):
    return render(request, 'admin_home.html')

def request_view(request):
    health_profiles = HealthProfile.objects.filter(user__status='inactive')
    context = {'health_profiles': health_profiles}
    return render(request, 'request.html', context)


def approve_health_center(request, pk):
    health_profile = HealthProfile.objects.get(pk=pk)
    user = health_profile.user
    user.status = 'active'
    user.save()
    # Send email to the user
    messages.success(request, f'Health center {health_profile.health_center_name} approved.')
    return redirect('requests')

@login_required
def health_home_view(request):
    User = get_user_model()  # Get the user model
    current_user = request.user

    # Check if the logged-in user is a healthcare provider
    if current_user.usertype == 'healthcare_provider':
        # Fetch the health center's name from first_name and last_name
        health_center_name = f"{current_user.first_name} {current_user.last_name}"
    else:
        health_center_name = "Health Center"  # Default value or handle as needed

    context = {
        'health_center_name': health_center_name,
    }
    return render(request, 'health_home.html', context)

def profile_view(request):
    try:
        user = request.user
        parent_profile = ParentProfile.objects.get(user=request.user)
        child_profile = ChildProfile.objects.get(parent=request.user)
        vaccinations = VaccinationRecord.objects.filter(child=child_profile)
        vaccines = []
        
        for vaccination in vaccinations:
            if vaccination.vaccine_taken:
                vaccines.append({
                    'name': vaccination.vaccine_name,
                    'date': vaccination.date,
                    'weight': vaccination.weight,
                    'place': vaccination.place
                })

        context = {
            'user': user,
            'parent_profile': parent_profile,
            'child_profile': child_profile,
            'vaccines': vaccines,
        }
        
        return render(request, 'parent_profile.html', context)
    except ParentProfile.DoesNotExist or ChildProfile.DoesNotExist:
        messages.error(request, 'Profile not found.')
        return redirect('home')

def edit_parentview(request):
    try:
        user = request.user
        parent_profile = ParentProfile.objects.get(user=user)
        child_profiles = ChildProfile.objects.filter(parent=user)

        if request.method == 'POST':
            # Update user's full name and email
            user.first_name = request.POST['parentName'].split(' ')[0]
            user.last_name = ' '.join(request.POST['parentName'].split(' ')[1:])
            user.email = request.POST['parentEmail']
            user.save()

            # Update ParentProfile
            parent_profile.contact_no = request.POST.get('contactNo', parent_profile.contact_no)
            parent_profile.parentno = request.POST.get('parentno', parent_profile.parentno)  # New field added
            parent_profile.address = request.POST.get('address', parent_profile.address)
            parent_profile.place = request.POST.get('place', parent_profile.place)
            parent_profile.save()

            # Update ChildProfile if applicable
            for child_profile in child_profiles:
                child_profile.child_name = request.POST.get('childName', child_profile.child_name)
                child_profile.dob = request.POST.get('dob') or child_profile.dob
                child_profile.gender = request.POST.get('gender', child_profile.gender)
                child_profile.blood_group = request.POST.get('bloodGroup', child_profile.blood_group)
                child_profile.birth_weight = request.POST.get('birthWeight') or child_profile.birth_weight
                child_profile.birth_height = request.POST.get('birthHeight') or child_profile.birth_height
                child_profile.age = request.POST.get('age') or child_profile.age
                child_profile.current_weight = request.POST.get('currentWeight') or child_profile.current_weight
                child_profile.current_height = request.POST.get('currentHeight') or child_profile.current_height
                child_profile.save()

                 # Update Vaccination Records
                vaccine_records = VaccinationRecord.objects.filter(child=child_profile)
                for index, vaccine in enumerate(vaccine_records):
                    vaccine.vaccine_name = request.POST.get(f'vaccine_name_{index + 1}', vaccine.vaccine_name)
                    vaccine.date = request.POST.get(f'vaccine_date_{index + 1}', vaccine.date)
                    vaccine.weight = request.POST.get(f'weight_{index + 1}', vaccine.weight)
                    vaccine.place = request.POST.get(f'vaccine_place_{index + 1}', vaccine.place)
                    vaccine.save()

            messages.success(request, 'Profile updated successfully.')
            return redirect('parent_profile')

        # Fetching vaccination records for the child profiles
        vaccines = []
        for child_profile in child_profiles:
            vaccine_records = VaccinationRecord.objects.filter(child=child_profile)
            for vaccine in vaccine_records:
                vaccines.append(vaccine)

        context = {
            'user': user,
            'parent_profile': parent_profile,
            'child_profile': child_profile,
            'vaccines': vaccines,
        }

        return render(request, 'edit_parentview.html', context)

    except ParentProfile.DoesNotExist:
        messages.error(request, 'Profile not found.')
        return redirect('home')

def total_parents(request, user_id=None):
    if user_id:
        # Handle POST request for a specific parent
        user = get_object_or_404(User, id=user_id)
        if request.method == 'POST':
            if user.status == 'active':
                user.status = 'inactive'
                messages.success(request, f'{user.email} has been deactivated.')
            else:
                user.status = 'active'
                messages.success(request, f'{user.email} has been activated.')
            
            user.save()
            return redirect('total_parents')  # Redirect to the page where you list the parents
        else:
            # For GET requests with a user_id, you could handle errors or redirect
            return redirect('total_parents')  # Redirect to the list of parents
    else:
        # Handle GET request to list all parents
        parents = User.objects.filter(usertype='parent')  # Filter by usertype=0 for parents
        return render(request, 'total_parents.html', {'parents': parents})

def total_healthcenters(request):
    # Filter users based on 'healthcare_provider' usertype
    healthcare_providers = User.objects.filter(usertype='healthcare_provider')
    return render(request, 'total_healthcenters.html', {'healthcare_providers': healthcare_providers})

def activate_healthcenter(request, id):
    healthcare_provider = get_object_or_404(User, id=id)
    if request.method == 'POST':
        if healthcare_provider.status == 'inactive':
            healthcare_provider.status = 'active'
        else:
            healthcare_provider.status = 'inactive'
        healthcare_provider.save()
    return redirect('total_healthcenters')

def add_vaccine(request):
    if request.method == 'POST':
        vaccine_name = request.POST.get('vaccine_name')
        manufacturer = request.POST.get('manufacturer')
        batch_number = request.POST.get('batch_number')
        date_manufacture = request.POST.get('date_manufacture')
        expiry_date = request.POST.get('expiry_date')
        age_group = request.POST.get('age_group')
        dose_number = request.POST.get('dose_number')
        interval_days = request.POST.get('interval_days')
        indications = request.POST.get('indications')
        stock = request.POST.get('stock')
        free_or_paid = request.POST.get('free_or_paid')
        rate = request.POST.get('rate') if free_or_paid == 'paid' else None

        # Validate required fields
        if not all([vaccine_name, manufacturer, batch_number, date_manufacture, expiry_date, age_group, dose_number, interval_days, indications, stock, free_or_paid]):
            messages.error(request, 'All fields except rate are required')
            return redirect('add_vaccine')

        try:
            # Create and save the vaccine entry
            vaccine = Vaccine(
                vaccine_name=vaccine_name,
                manufacturer=manufacturer,
                batch_number=batch_number,
                date_manufacture=date_manufacture,
                expiry_date=expiry_date,
                age_group=age_group,
                indications=indications,
                stock=stock,
                free_or_paid=free_or_paid,
                rate=rate
            )
            vaccine.save()

            # Create and save the dose entry
            vaccine_dose = VaccineDose(
                vaccine=vaccine,  # foreign key to the Vaccine table
                dose_number=dose_number,
                interval_days=interval_days
            )
            vaccine_dose.save()

            messages.success(request, 'Vaccine details added successfully')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')

    return render(request, 'add_vaccine.html')


def view_vaccines(request):
    vaccines = Vaccine.objects.all()
    return render(request, 'view_vaccines.html', {'vaccines': vaccines})


def delete_vaccine(request, vaccine_id):
    vaccine = get_object_or_404(Vaccine, vaccine_id=vaccine_id)
    VaccineDose.objects.filter(vaccine=vaccine).delete()
    vaccine.delete()
    return redirect('view_vaccines')


@login_required
def add_vaccine_request(request):
    if request.method == 'POST':
        vaccine_id = request.POST.get('vaccine')
        dose_id = request.POST.get('dose')
        requested_stock = request.POST.get('stock')
        healthcenter = HealthProfile.objects.get(user=request.user)

        # Use vaccine_id instead of id
        vaccine = Vaccine.objects.get(vaccine_id=vaccine_id)
        dose = VaccineDose.objects.get(dose_id=dose_id)

        # Save the VaccineRequest
        vaccine_request = VaccineRequest.objects.create(
            healthcenter=healthcenter,
            vaccine=vaccine,
            dose=dose,
            requested_stock=requested_stock,
            status='Pending',
            request_date=datetime.now(),
        )

        # Save to VaccineRequestHistory
        VaccineRequestHistory.objects.create(
            healthcenter=healthcenter,
            vaccine=vaccine,
            dose=dose,
            requested_stock=requested_stock,
            status='Pending',
            request_date=datetime.now(),
        )

        return redirect('vaccine_request_success')  # Redirect to a success page or the same page with a success message

    vaccines = Vaccine.objects.all()
    return render(request, 'addvaccine_req.html', {'vaccines': vaccines})


def load_doses(request):
    vaccine_id = request.GET.get('vaccine_id')
    if vaccine_id:
        doses = VaccineDose.objects.filter(vaccine_id=vaccine_id).values('dose_id', 'dose_number')
    else:
        doses = []

    return JsonResponse(list(doses), safe=False)

@login_required
def vaccine_request_success(request):
    # Get the health center associated with the logged-in user
    healthcenter = get_object_or_404(HealthProfile, user=request.user)
    
    # Retrieve all vaccine requests for this health center
    vaccine_requests = VaccineRequest.objects.filter(healthcenter=healthcenter)

    context = {
        'vaccine_requests': vaccine_requests,
        'message': 'Vaccine request was successful!'
    }
    return render(request, 'vaccine_request_success.html', context)

@login_required
def delete_vaccine_request(request, request_id):
    # Get the vaccine request object
    vaccine_request = get_object_or_404(VaccineRequest, id=request_id)
    
    # Check if the request is from the logged-in health center
    if vaccine_request.healthcenter.user == request.user:
        # Delete the vaccine request
        vaccine_request.delete()
    
    # Redirect to the success page or back to the list of requests
    return redirect('vaccine_request_success')

def view_vaccine_details(request, vaccine_id):
    vaccine = get_object_or_404(Vaccine, vaccine_id=vaccine_id)
    doses = VaccineDose.objects.filter(vaccine=vaccine)

    context = {
        'vaccine': vaccine,
        'doses': doses
    }
    return render(request, 'view_vaccine_details.html', context)

@login_required
def vaccinereq_view(request):
    vaccine_requests = VaccineRequest.objects.all()
    return render(request, 'vaccinereq.html', {'vaccine_requests': vaccine_requests})

@login_required
def approve_vaccine_request(request, request_id):
    # Fetch the vaccine request
    vaccine_request = get_object_or_404(VaccineRequest, id=request_id)
    
    # Update the vaccine stock
    vaccine = vaccine_request.vaccine
    vaccine.stock = max(0, vaccine.stock - vaccine_request.requested_stock)  # Ensure stock doesn't go negative
    vaccine.save()

    # Update the request status and approval date
    vaccine_request.status = 'Approved'
    vaccine_request.approval_date = timezone.now()
    vaccine_request.save()

    # Create a record in the request history
    VaccineRequestHistory.objects.create(
        healthcenter=vaccine_request.healthcenter,
        vaccine=vaccine_request.vaccine,
        dose=vaccine_request.dose,
        requested_stock=vaccine_request.requested_stock,
        status='Approved',
        approval_date=vaccine_request.approval_date  # Use the same approval date
    )
    
    return redirect('vaccine_requests')

@login_required
def reject_vaccine_request(request, request_id):
    # Fetch the vaccine request
    vaccine_request = get_object_or_404(VaccineRequest, id=request_id)

    # Update the request status
    vaccine_request.status = 'Rejected'
    vaccine_request.save()

    # Create a record in the request history
    VaccineRequestHistory.objects.create(
        healthcenter=vaccine_request.healthcenter,
        vaccine=vaccine_request.vaccine,
        dose=vaccine_request.dose,
        requested_stock=vaccine_request.requested_stock,
        status='Rejected',
        approval_date=timezone.now()  # Set the rejection time
    )
    
    return redirect('vaccinereq')

def select_vaccine(request):
    vaccine = Vaccine.objects.all()
    return render(request, 'select_vaccine.html', {'vaccines': vaccine})


@login_required
def select_healthcenter(request, vaccine_id):
    # Fetching vaccine details
    vaccine = get_object_or_404(Vaccine, pk=vaccine_id)
    
    # Fetch health profiles that have the selected vaccine
    healthprofiles = HealthProfile.objects.filter(
        id__in=VaccineRequest.objects.filter(vaccine_id=vaccine_id).values_list('healthcenter_id', flat=True)
    )
    
    context = {
        'vaccine_id': vaccine_id,
        'healthprofiles': healthprofiles
    }
    return render(request, 'select_healthcenter.html', context)


logger = logging.getLogger(__name__)

@login_required
def schedule_appointment(request):
    if request.method == 'POST':
        vaccine_id = request.POST.get('vaccine_id')
        healthcenter_id = request.POST.get('healthcenter_id')
        appointment_date_str = request.POST.get('appointment_date')
        appointment_time_str = request.POST.get('appointment_time')

        # Validate form data
        if not vaccine_id or not healthcenter_id or not appointment_date_str or not appointment_time_str:
            messages.error(request, "Error: All fields are required.")
            return redirect('schedule_appointment')  # Redirect to the scheduling page

        try:
            vaccine = get_object_or_404(Vaccine, pk=vaccine_id)
            healthcenter = get_object_or_404(HealthProfile, pk=healthcenter_id)
            appointment_date = datetime.strptime(appointment_date_str, '%Y-%m-%d').date()
            appointment_time = datetime.strptime(appointment_time_str.split('-')[0], '%H:%M').time()

            # Check if the selected slot is already taken
            if Appointment.objects.filter(
                health_center=healthcenter,
                appointment_date=appointment_date,
                appointment_time=appointment_time
            ).exists():
                messages.error(request, "Error: The selected date and time slot is already taken.")
                return render(request, 'schedule.html', {
                    'vaccine': vaccine,
                    'healthcenter': healthcenter,
                    'parent_profile': get_object_or_404(ParentProfile, user=request.user),
                    'child_profile': get_object_or_404(ChildProfile, parent=request.user),
                })

        except (ValueError, ObjectDoesNotExist) as e:
            logger.error(f"Error: {e}")
            messages.error(request, "Error: Invalid data format or object not found.")
            return redirect('schedule_appointment')

        user = request.user

        # Create and save the appointment
        try:
            appointment = Appointment(
                vaccine=vaccine,
                health_center=healthcenter,
                user=user,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                status='Pending'  # Set initial status
            )
            appointment.full_clean()  # Validate fields
            appointment.save()

            # # Create a notification
            # notification_message = f"New appointment scheduled for {appointment.appointment_date} at {appointment.appointment_time}"
            # Notification.objects.create(user=request.user, message=notification_message)

            # Send email
            subject = 'Appointment Scheduled'
            html_message = render_to_string('appointment_scheduled_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )

            # Use a custom message storage
            request.session['appointment_success'] = 'Appointment scheduled successfully!'
            return redirect('appointment_success')

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            # Use a custom message storage
            request.session['appointment_error'] = f'An unexpected error occurred: {str(e)}'
            return redirect('schedule_appointment')

    else:  # GET request
        vaccine_id = request.GET.get('vaccine_id')
        healthcenter_id = request.GET.get('healthcenter_id')

        if not vaccine_id or not healthcenter_id:
            messages.error(request, "Error: Missing vaccine_id or healthcenter_id.")
            return redirect('select_vaccine')  # Redirect to vaccine selection page

        try:
            vaccine = get_object_or_404(Vaccine, pk=vaccine_id)
            healthcenter = get_object_or_404(HealthProfile, pk=healthcenter_id)
            user = request.user
            parent_profile = get_object_or_404(ParentProfile, user=user)
            child_profile = get_object_or_404(ChildProfile, parent=user)

            context = {
                'vaccine': vaccine,
                'healthcenter': healthcenter,
                'parent_profile': parent_profile,
                'child_profile': child_profile,
            }
            return render(request, 'schedule.html', context)
        except Exception as e:
            logger.error(f"Error in GET request: {e}")
            messages.error(request, f'An error occurred: {str(e)}')
            return redirect('select_vaccine')  # Redirect to vaccine selection page

@login_required
def appointment_success(request):
    appointments = Appointment.objects.filter(user=request.user).order_by('-appointment_date', '-appointment_time')
    success_message = request.session.pop('appointment_success', None)
    context = {
        'appointments': appointments,
        'success_message': success_message
    }
    return render(request, 'appointment_success.html', context)

@login_required
def manage_appointments(request):
    try:
        health_center = HealthProfile.objects.get(user=request.user)
    except HealthProfile.DoesNotExist:
        request.session['health_center_error'] = "You are not associated with a health center."
        return redirect('home')

    appointments = Appointment.objects.filter(health_center=health_center).order_by('appointment_date', 'appointment_time')
    success_message = request.session.pop('appointment_status_success', None)
    context = {
        'appointments': appointments,
        'success_message': success_message
    }
    return render(request, 'manage_appointments.html', context)

@login_required
def update_appointment_status(request, appointment_id):
    if request.method == 'POST':
        appointment = get_object_or_404(Appointment, id=appointment_id)
        action = request.POST.get('action')
        
        if action == 'approve':
            appointment.status = 'Approved'
            appointment.approval_date = timezone.now()
            appointment.save()
            
            # Send approval email
            subject = 'Appointment Approved'
            html_message = render_to_string('appointment_approved_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )
            
            request.session['appointment_status_success'] = f"Appointment for {appointment.user.username} on {appointment.appointment_date} has been approved."
        elif action == 'reject':
            appointment.status = 'Rejected'
            appointment.save()
            
            # Send rejection email
            subject = 'Appointment Rejected'
            html_message = render_to_string('appointment_rejected_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )
            
            request.session['appointment_status_success'] = f"Appointment for {appointment.user.username} on {appointment.appointment_date} has been rejected."
        
        appointment.updated_at = timezone.now()
        appointment.save()
    
    return redirect('manage_appointments')

@login_required
def appointment_success(request):
    now = timezone.now()  # Get the current time with timezone awareness
    logging.info(f"Current time: {now}")

    # Fetch appointments for the user, ordered by date and time
    appointments = Appointment.objects.filter(user=request.user).order_by('-appointment_date', '-appointment_time')
    logging.info(f"Found {appointments.count()} appointments.")

    for appointment in appointments:
        # Combine date and time into a single timezone-aware datetime object
        appointment_datetime = timezone.make_aware(
            datetime.combine(appointment.appointment_date, appointment.appointment_time),
            timezone.get_current_timezone()
        )

        logging.info(f"Checking appointment {appointment.id}: Current Status - {appointment.status}, DateTime - {appointment_datetime}")

        # Check if the appointment has passed and is not already marked "Completed"
        if appointment.status != 'Completed' and appointment_datetime <= now:
            logging.info(f"Appointment {appointment.id} is due for completion.")
            
            # Update appointment status
            old_status = appointment.status
            appointment.status = 'Completed'
            appointment.save()
            logging.info(f"Appointment {appointment.id} status changed from {old_status} to {appointment.status}")

            # Verify the status change
            updated_appointment = Appointment.objects.get(id=appointment.id)
            logging.info(f"Verified status for appointment {appointment.id}: {updated_appointment.status}")

            # Update VaccineRequest stock for the health center
            vaccine_request = VaccineRequest.objects.filter(
                healthcenter=appointment.health_center,
                vaccine=appointment.vaccine
            ).first()

            if vaccine_request:
                old_stock = vaccine_request.requested_stock
                if old_stock > 0:
                    vaccine_request.requested_stock = max(0, old_stock - 1)  # Decrease stock by 1
                    vaccine_request.save()
                    logging.info(f"VaccineRequest {vaccine_request.id} stock reduced from {old_stock} to {vaccine_request.requested_stock}")
                else:
                    logging.warning(f"VaccineRequest {vaccine_request.id} stock is already 0, cannot reduce further.")

                # Verify the stock change
                updated_vaccine_request = VaccineRequest.objects.get(id=vaccine_request.id)
                logging.info(f"Verified stock for VaccineRequest {vaccine_request.id}: {updated_vaccine_request.requested_stock}")
            else:
                logging.warning(f"No VaccineRequest found for appointment {appointment.id}")

            # Optionally send email notification and create Notification object here
            # ...

    # Refresh the appointments queryset to get updated statuses
    updated_appointments = Appointment.objects.filter(user=request.user).order_by('-appointment_date', '-appointment_time')
    
    success_message = request.session.pop('appointment_success', None)
    
    context = {
        'appointments': updated_appointments,
        'success_message': success_message,
    }
    
    return render(request, 'appointment_success.html', context)

@login_required
def notification_view(request):
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'notification.html', {'notifications': notifications})
    
def update_appointment_status(request, appointment_id):
    if request.method == 'POST':
        appointment = get_object_or_404(Appointment, id=appointment_id)
        action = request.POST.get('action')
        
        if action == 'approve':
            appointment.status = 'Approved'
            appointment.approval_date = timezone.now()
            appointment.save()
            
            # Send approval email
            subject = 'Appointment Approved'
            html_message = render_to_string('appointment_approved_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )
            
            request.session['appointment_status_success'] = f"Appointment for {appointment.user.username} on {appointment.appointment_date} has been approved."
        elif action == 'reject':
            appointment.status = 'Rejected'
            appointment.save()
            
            # Send rejection email
            subject = 'Appointment Rejected'
            html_message = render_to_string('appointment_rejected_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
                'reason': request.POST.get('rejection_reason', 'No reason provided'),
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )
            
            request.session['appointment_status_success'] = f"Appointment for {appointment.user.username} on {appointment.appointment_date} has been rejected."
        
        appointment.updated_at = timezone.now()
        appointment.save()
    
    return redirect('manage_appointments')

# @login_required
# def appointment_success(request):
#     appointments = Appointment.objects.filter(user=request.user).order_by('-appointment_date', '-appointment_time')
#     success_message = request.session.pop('appointment_success', None)
#     context = {
#         'appointments': appointments,
#         'success_message': success_message
#     }
#     return render(request, 'appointment_success.html', context)

@login_required
def delete_appointment(request, appointment_id):
    if request.method == 'POST':
        appointment = get_object_or_404(Appointment, id=appointment_id, user=request.user)
        appointment.delete()
        messages.success(request, 'Appointment deleted successfully.')
    return redirect('appointment_success')

@login_required
def manage_appointments(request):
    # Check if the user is associated with a health center
    try:
        health_center = HealthProfile.objects.get(user=request.user)
    except HealthProfile.DoesNotExist:
        messages.error(request, "You are not associated with a health center.")
        return redirect('home')  # or wherever you want to redirect non-health center users

    appointments = Appointment.objects.filter(health_center=health_center).order_by('appointment_date', 'appointment_time')
    
    success_message = request.session.pop('appointment_status_success', None)
    context = {
        'appointments': appointments,
        'success_message': success_message
    }
    return render(request, 'manage_appointments.html', context)


@login_required
def update_appointment_status(request, appointment_id):
    if request.method == 'POST':
        appointment = get_object_or_404(Appointment, id=appointment_id)
        action = request.POST.get('action')
        
        if action == 'approve':
            appointment.status = 'Approved'
            appointment.approval_date = timezone.now()
            appointment.save()
            
            # Send approval email
            subject = 'Appointment Approved'
            html_message = render_to_string('appointment_approved_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )
            
            # Create a notification
            Notification.objects.create(
                user=appointment.user,
                message=f"Your appointment on {appointment.appointment_date} has been approved.",
                related_appointment=appointment
            )
            
            request.session['appointment_status_success'] = f"Appointment for {appointment.user.username} on {appointment.appointment_date} has been approved."
        elif action == 'reject':
            appointment.status = 'Rejected'
            appointment.save()
            
            # Send rejection email
            subject = 'Appointment Rejected'
            html_message = render_to_string('appointment_rejected_email.html', {
                'appointment': appointment,
                'user': appointment.user,
                'health_center': appointment.health_center,
                'reason': request.POST.get('rejection_reason', 'No reason provided'),
            })
            plain_message = strip_tags(html_message)
            send_mail(
                subject,
                plain_message,
                'nurturenest02@example.com',
                [appointment.user.email],
                html_message=html_message,
            )
            
            # Create a notification for rejection
            Notification.objects.create(
                user=appointment.user,
                message=f"Your appointment on {appointment.appointment_date} has been rejected.",
                related_appointment=appointment
            )
            
            request.session['appointment_status_success'] = f"Appointment for {appointment.user.username} on {appointment.appointment_date} has been rejected."
        
        appointment.updated_at = timezone.now()
        appointment.save()
    
    return redirect('manage_appointments')    

User = get_user_model()

def forgot_password(request):
    if request.method == 'POST':
        email = request.POST['email']
        try:
            user = User.objects.get(email=email)
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            reset_link = request.build_absolute_uri(reverse('reset_password', kwargs={'uidb64': uid, 'token': token}))

            context = {
                'user': user,
                'reset_link': reset_link,
            }

            message = render_to_string('reset_password_email.html', context)

            send_mail(
                'Password Reset Request',
                message,
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
                html_message=message,
            )

            return render(request, 'forgot_password.html', {'message': 'A reset link has been sent to your email address.'})
        except User.DoesNotExist:
            return render(request, 'forgot_password.html', {'error': 'Email does not exist'})
    return render(request, 'forgot_password.html')

def reset_password(request, uidb64, token):
    if request.method == 'POST':
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        if password == confirm_password:
            try:
                uid = force_str(urlsafe_base64_decode(uidb64))
                user = User.objects.get(pk=uid)
                if default_token_generator.check_token(user, token):
                    user.set_password(password)
                    user.save()
                    return render(request, 'reset_password_complete.html')
                else:
                    return render(request, 'reset_password_confirm.html', {'error': 'Invalid token'})
            except (TypeError, ValueError, OverflowError, User.DoesNotExist):
                return render(request, 'reset_password_confirm.html', {'error': 'Invalid link'})
        else:
            return render(request, 'reset_password_confirm.html', {'error': 'Passwords do not match'})
    return render(request, 'reset_password_confirm.html', {'uidb64': uidb64, 'token': token})

def notification(request):
    return render(request, 'notification.html')

def unread_notifications(request):
    if request.user.is_authenticated:
        unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
        return {'unread_notifications_count': unread_count}
    return {'unread_notifications_count': 0}

@login_required
def notification_view(request):
    # Fetch notifications for the logged-in user
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    # Optionally mark notifications as read when viewed
    notifications.update(is_read=True)  # This marks all notifications as read when viewed
    
    # Count unread notifications for the navbar
    unread_notifications_count = Notification.objects.filter(user=request.user, is_read=False).count()
    
    return render(request, 'notification.html', {
        'notifications': notifications,
        'unread_notifications_count': unread_notifications_count,
    })

@login_required
def delete_notification(request, notification_id):
    if request.method == 'POST':
        notification = get_object_or_404(Notification, id=notification_id)
        notification.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False}, status=400)
    
def health_profile_view(request):
    try:
        user = request.user
        health_profile = HealthProfile.objects.get(user=user)
        context = {
            'user': user,
            'health_profile': health_profile,
        }
        return render(request, 'view_healthprofile.html', context)

    except HealthProfile.DoesNotExist:
        messages.error(request, 'Health profile not found.')
        return redirect('health_home') 

@login_required
def edit_health_profile_view(request):
    health_profile = get_object_or_404(HealthProfile, user=request.user)
    
    if request.method == 'POST':
        # Update the health profile
        health_profile.health_center_name = request.POST.get('health_center_name')
        health_profile.phone = request.POST.get('phone')
        health_profile.address = request.POST.get('address')
        health_profile.city = request.POST.get('city')
        health_profile.license_number = request.POST.get('license_number')
        
        health_profile.save()
        
        messages.success(request, 'Health profile updated successfully.')
        return redirect('view_health_profile')
    
    context = {
        'health_profile': health_profile,
        'user': request.user,
    }
    return render(request, 'edit_healthprofile.html', context)
    


def chart_view(request):
    user = request.user

    # Fetch vaccination records for the logged-in user's child
    vaccination_records = VaccinationRecord.objects.filter(child__parent=user)

    # Fetch completed appointments for the logged-in user
    appointments = Appointment.objects.filter(user=user, status='completed')

    # Get the user's child profile and calculate the current age in months
    child_profile = ChildProfile.objects.filter(parent=user).first()
    
    if child_profile:
        current_age_in_months = calculate_age_in_months(child_profile.dob)

        # Get all vaccines from the Vaccine table
        vaccines = Vaccine.objects.all()

        # Convert age group into months for sorting purposes
        def convert_to_weeks_months_years(age_group):
            try:
                if 'week' in age_group:
                    return int(age_group.split()[0]) / 4  # Convert weeks to months
                elif 'month' in age_group:
                    return int(age_group.split()[0])  # Already in months
                elif 'year' in age_group:
                    return int(age_group.split()[0]) * 12  # Convert years to months
            except (ValueError, IndexError):
                return float('inf')  # For invalid or unknown age groups, return infinity

        # Sort all vaccines by age group (first weeks, then months, then years)
        sorted_vaccines = sorted(vaccines, key=lambda vaccine: convert_to_weeks_months_years(vaccine.age_group))

        # Determine which vaccines are completed
        completed_vaccines = set()

        # Mark default vaccines (age group "1 week") as completed
        for vaccine in sorted_vaccines:
            if vaccine.age_group == "1 week":
                completed_vaccines.add(vaccine.vaccine_name)
        
        # Add vaccines from VaccinationRecord to completed set
        for record in vaccination_records:
            completed_vaccines.add(record.vaccine_name)
        
        # Add vaccines from completed appointments to the completed set
        for appointment in appointments:
            completed_vaccines.add(appointment.vaccine_name)

        # Find the latest completed vaccine age from the completed set
        latest_completed_vaccine_age = 0
        for vaccine in sorted_vaccines:
            vaccine_age_in_months = convert_to_weeks_months_years(vaccine.age_group)
            if vaccine.vaccine_name in completed_vaccines:
                latest_completed_vaccine_age = max(latest_completed_vaccine_age, vaccine_age_in_months)

        # Mark vaccines as completed or upcoming based on age and completion status
        for vaccine in sorted_vaccines:
            vaccine_age_in_months = convert_to_weeks_months_years(vaccine.age_group)

            # If the vaccine is in the completed set or within the current age, mark it as completed
            if vaccine.vaccine_name in completed_vaccines or vaccine_age_in_months <= current_age_in_months:
                vaccine.completed = True
            else:
                vaccine.completed = False

            # Check for upcoming vaccines (within 1 month) that have not been completed yet
            upcoming_vaccine_threshold = 1  # Define threshold as 1 month before the vaccine age
            if vaccine.vaccine_name not in completed_vaccines and \
               current_age_in_months < vaccine_age_in_months and \
               vaccine_age_in_months - current_age_in_months <= upcoming_vaccine_threshold:
                
                # Check if a notification for this vaccine is already sent in the current session
                session_key = f"vaccine_reminder_{vaccine.vaccine_name}"
                if not request.session.get(session_key, False):
                    # Send a notification to the user (save it to DB if necessary)
                    notification_message = f"Reminder: Your child is nearing the recommended age for {vaccine.vaccine_name}. Please schedule an appointment."
                    Notification.objects.create(
                        user=user,
                        message=notification_message,
                        related_appointment=None
                    )
                    
                    # Store the reminder in the session to prevent duplicate notifications
                    request.session[session_key] = True

    context = {
        'vaccines': sorted_vaccines,
        'completed_vaccines': completed_vaccines,
    }

    return render(request, 'chart.html', context)



def calculate_age_in_months(birth_date):
    # Get the current date
    today = date.today()
    
    # Calculate the difference in years
    years_difference = today.year - birth_date.year
    
    # Calculate the difference in months
    months_difference = today.month - birth_date.month
    
    # Combine the difference in years and months
    age_in_months = (years_difference * 12) + months_difference
    
    return age_in_months


@receiver(user_logged_out)
def clear_notifications(sender, request, user, **kwargs):
    # Expire the session entirely upon logout
    request.session.flush()

    # Clear all vaccine reminder notifications for the user
    Notification.objects.filter(user=user, message__icontains='Reminder:').delete()


# View for image upload and vaccine prediction
def upload_image(request):
    if request.method == 'POST':
        # Get the uploaded image
        image = request.FILES['vaccine_image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'vaccine_images', image.name)  # Save in 'vaccine_images' folder
        
        # Ensure the 'vaccine_images' directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save the image to the media folder
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        # Construct the URL to access the image (relative to MEDIA_URL)
        image_url = os.path.join(settings.MEDIA_URL, 'vaccine_images', image.name)
        
        # Call the ML model to predict vaccine details
        vaccine_details = predict_vaccine_details(image_path)
        
        # Return the predicted details in JSON format including the image URL
        return JsonResponse({
            'vaccine_name': vaccine_details['name'],
            'age_group': vaccine_details['age_group'],
            'purpose': vaccine_details['purpose'],
            'disadvantages': vaccine_details['disadvantages'],
            'image_url': image_url  # Pass the image URL for display
        })
    
    return render(request, 'upload_image.html')

def delete_healthcenter(request, healthcare_provider_id):
    if request.method == 'POST':
        healthcare_provider = get_object_or_404(User, id=healthcare_provider_id)
        healthcare_provider.delete()  # Delete the healthcare provider
        messages.success(request, 'HealthCenter deleted successfully.')
    return redirect('total_healthcenters')  # Redirect to the health centers list page

def delete_parent(request, id):
    parent = get_object_or_404(User, id=id)
    parent.delete()  # Delete the user
    return redirect('total_parents')  # Redirect back to the parent list after deletion

def add_feedingchart(request):
    if request.method == 'POST':
        age = request.POST.get('age')
        main_heading = request.POST.get('main_heading')
        description = request.POST.get('description')

        # Basic validation (you can expand this as needed)
        if not age or not main_heading or not description:
            messages.error(request, "All fields are required.")
            return render(request, 'add_feedingchart.html')

        # Save the form data to the database
        FeedingChart.objects.create(
            age=age,
            main_heading=main_heading,
            description=description
        )
        
        messages.success(request, "Feeding chart added successfully!")
        return redirect('add_feedingchart')

    return render(request, 'add_feedingchart.html')

# View to list all feeding charts
def feedingchart_lists(request):
    feedingcharts = FeedingChart.objects.all()
    return render(request, 'feedingchart_lists.html', {'feedingcharts': feedingcharts})

# View to show details of a single feeding chart
def feedingchart_details(request, chart_id):
    feedingchart = get_object_or_404(FeedingChart, id=chart_id)
    return render(request, 'feedingchart_details.html', {'feedingchart': feedingchart})

def view_feedingchart(request):
    # Fetch all feeding chart data from the database
    feedingcharts = FeedingChart.objects.all()
    context = {
        'feedingcharts': feedingcharts,
    }
    
    return render(request, 'view_feedingchart.html', context)

def add_mentalhealth(request):
    if request.method == 'POST':
        # Retrieving the posted data
        age = request.POST.get('age')
        descriptions = request.POST.getlist('description[]')
        image = request.FILES.get('image')

        # Validate inputs
        if not age:
            messages.error(request, 'Age is required.')
            return redirect('add_mentalhealth')

        if not descriptions or any(desc == "" for desc in descriptions):
            messages.error(request, 'All description fields must be filled.')
            return redirect('add_mentalhealth')

        if not image:
            messages.error(request, 'An image upload is required.')
            return redirect('add_mentalhealth')

        # Save the uploaded image
        fs = FileSystemStorage()
        image_name = fs.save(image.name, image)

        # Create the main MentalHealthDetails record (age and image)
        mental_health_record = MentalHealthDetails(age=age, image=image_name)
        mental_health_record.save()

        # Save each description as a separate entry linked to the same MentalHealthDetails
        for desc in descriptions:
            description_record = MentalHealthDescription(mental_health_detail=mental_health_record, description=desc)
            description_record.save()

        # Success message and redirect
        messages.success(request, 'Mental health details added successfully!')
        return redirect('add_mentalhealth')

    return render(request, 'add_mentalhealth.html')

def mentalhealth_lists(request):
    mental_health_details = MentalHealthDetails.objects.all()  # Fetch all records
    return render(request, 'mentalhealth_lists.html', {'mental_health_details': mental_health_details})

def mentalhealth_listsdetails(request, mental_health_id):
    # Get the mental health details record by its ID
    mental_health = get_object_or_404(MentalHealthDetails, id=mental_health_id)
    return render(request, 'mentalhealth_listsdetails.html', {'mental_health': mental_health})

def delete_mentalhealth(request, id):
    mental_health = get_object_or_404(MentalHealthDetails, id=id)
    mental_health.delete()  # Delete the record
    return redirect(reverse('mentalhealth_lists')) 

# View to list all mental health details
def view_mentalhealth(request):
    # Fetch all mental health details from the database
    mental_health_details = MentalHealthDetails.objects.all()
    return render(request, 'view_mentalhealth.html', {'mental_health_details': mental_health_details})

# View to display full details for a specific mental health entry
def view_mentalhealthdetails(request, pk):
    # Fetch the mental health detail by ID (primary key)
    mental_health_detail = get_object_or_404(MentalHealthDetails, pk=pk)
    return render(request, 'view_mentalhealthdetails.html', {'view_mentalhealthdetails': mental_health_detail})

def vaccination_history(request):
    parent = request.user
    children = ChildProfile.objects.filter(parent=parent)

    vaccination_data = []

    for child in children:
        today = datetime.today().date()
        child_age_in_months = (today.year - child.dob.year) * 12 + (today.month - child.dob.month)

        vaccines = Vaccine.objects.all()

        def convert_to_weeks_months_years(age_group):
            try:
                if 'week' in age_group:
                    return int(age_group.split()[0]) / 4
                elif 'month' in age_group:
                    return int(age_group.split()[0])
                elif 'year' in age_group:
                    return int(age_group.split()[0]) * 12
            except (ValueError, IndexError):
                return float('inf')

        sorted_vaccines = sorted(vaccines, key=lambda vaccine: convert_to_weeks_months_years(vaccine.age_group))

        vaccination_records = VaccinationRecord.objects.filter(child=child)
        completed_appointments = Appointment.objects.filter(
            user=parent, status='Completed'
        )

        completed_vaccines = {}
        
        for vaccine in sorted_vaccines:
            if vaccine.age_group == "1 week":
                completed_vaccines[vaccine.vaccine_name] = {"place": "Default", "date": None}

        for record in vaccination_records:
            completed_vaccines[record.vaccine_name] = {"place": record.place, "date": record.date}

        for appointment in completed_appointments:
            completed_vaccines[appointment.vaccine.vaccine_name] = {
                "place": appointment.health_center.health_center_name,
                "date": appointment.appointment_date
            }

        child_vaccination_data = []

        for vaccine in sorted_vaccines:
            vaccine_age_in_months = convert_to_weeks_months_years(vaccine.age_group)
            is_completed = vaccine.vaccine_name in completed_vaccines or vaccine_age_in_months <= child_age_in_months

            if is_completed:
                vaccine_info = completed_vaccines.get(vaccine.vaccine_name, {"place": "N/A", "date": None})
                child_vaccination_data.append({
                    'vaccine': vaccine,
                    'is_completed': is_completed,
                    'place': vaccine_info["place"],
                    'date': vaccine_info["date"],
                })

        vaccination_data.append({
            'child': child,
            'child_vaccination_data': child_vaccination_data,
        })

    context = {
        'vaccination_data': vaccination_data
    }

    return render(request, 'vaccination_history.html', context)


def ecom_admin_home(request):
    return render(request, 'ecom_admin_home.html')

def add_category(request):
    if request.method == 'POST':
        category_name = request.POST.get('category_name', '').strip()
        if category_name:  # Ensure input is not empty
            try:
                # Create and save the new category
                category = Category.objects.create(name=category_name)
                category.save()
                messages.success(request, f'Category "{category_name}" added successfully!')
                return redirect('add_category')  # Redirect to the same page
            except Exception as e:
                messages.error(request, f'Error: {str(e)}')
        else:
            messages.error(request, "Category name cannot be empty.")
    
    return render(request, 'add_category.html')

def view_categories(request):
    try:
        categories = Category.objects.all()  # Retrieve all categories
    except Category.DoesNotExist:
        categories = []  # In case no categories exist
    
    return render(request, 'view_categories.html', {'categories': categories})

def edit_category(request, category_id):
    category = get_object_or_404(Category, id=category_id)
    if request.method == 'POST':
        category_name = request.POST.get('category_name', '').strip()
        if category_name:
            category.name = category_name
            category.save()
            messages.success(request, f'Category "{category_name}" updated successfully!')
            return redirect('view_categories')
        else:
            messages.error(request, "Category name cannot be empty.")
    
    return render(request, 'edit_category.html', {'category': category})

def delete_category(request, category_id):
    category = get_object_or_404(Category, id=category_id)
    if request.method == 'POST':
        category.delete()
        messages.success(request, f'Category "{category.name}" deleted successfully!')
        return redirect('view_categories')
    return redirect('view_categories')

@login_required
def add_product(request):
    if request.method == 'POST':
        try:
            with transaction.atomic():
                # Create Product instance
                product = Product.objects.create(
                    category_id=request.POST.get('category'),
                    product_name=request.POST.get('product_name'),
                    price=request.POST.get('price'),
                    stock=request.POST.get('stock')
                )

                # Handle multiple descriptions
                description_titles = request.POST.getlist('description_titles[]')
                descriptions = request.POST.getlist('descriptions[]')
                
                for title, desc in zip(description_titles, descriptions):
                    ProductDescription.objects.create(
                        product=product,
                        title=title,
                        description=desc
                    )

                # Handle multiple images
                images = request.FILES.getlist('images[]')
                for image in images:
                    ProductImage.objects.create(
                        product=product,
                        image=image
                    )

                messages.success(request, 'Product added successfully!')
                return redirect('add_product')

        except Exception as e:
            messages.error(request, f'Error adding product: {str(e)}')
            return redirect('add_product')

    categories = Category.objects.all()
    context = {
        'categories': categories,
        'page_title': 'Add Product'
    }
    return render(request, 'add_product.html', context)

@login_required
def view_addproducts(request):
    """View to display all products in a table format"""
    try:
        # Get all products with their related data, ordered by most recent first
        products = Product.objects.select_related('category').prefetch_related(
            'images', 
            'descriptions'
        ).order_by('-created_at')
        
        context = {
            'products': products,
            'page_title': 'View Products'
        }
        return render(request, 'view_addproducts.html', context)
    except Exception as e:
        messages.error(request, f'Error loading products: {str(e)}')
        return redirect('ecom_admin_home')
    
@login_required
def view_moredet(request, product_id):
    """View to display detailed information about a specific product"""
    try:
        product = get_object_or_404(
            Product.objects.select_related('category').prefetch_related(
                'images',
                'descriptions'
            ),
            id=product_id
        )
        
        context = {
            'product': product,
            'page_title': f'Product Details - {product.product_name}'
        }
        return render(request, 'view_moredet.html', context)
    except Exception as e:
        messages.error(request, f'Error loading product details: {str(e)}')
        return redirect('view_addproducts')
    
@login_required
def edit_productdet(request, product_id):
    """View to edit an existing product"""
    try:
        product = get_object_or_404(Product, id=product_id)
        categories = Category.objects.all()

        if request.method == 'POST':
            try:
                with transaction.atomic():
                    # Update basic information
                    product.category_id = request.POST.get('category')
                    product.product_name = request.POST.get('product_name')
                    product.price = request.POST.get('price')
                    product.stock = request.POST.get('stock')
                    product.save()

                    # Handle descriptions
                    product.descriptions.all().delete()
                    description_titles = request.POST.getlist('description_titles[]')
                    descriptions = request.POST.getlist('descriptions[]')
                    for title, desc in zip(description_titles, descriptions):
                        if title and desc:
                            ProductDescription.objects.create(
                                product=product,
                                title=title,
                                description=desc
                            )

                    # Handle new images
                    new_images = request.FILES.getlist('new_images[]')
                    for image in new_images:
                        ProductImage.objects.create(
                            product=product,
                            image=image
                        )

                messages.success(request, 'Product updated successfully!')
                return redirect('view_addproducts')
            
            except Exception as e:
                messages.error(request, f'Error updating product: {str(e)}')
                return redirect('view_addproducts')

        context = {
            'product': product,
            'categories': categories,
        }
        return render(request, 'edit_productdet.html', context)

    except Exception as e:
        messages.error(request, f'Error loading product: {str(e)}')
        return redirect('view_addproducts')

@login_required
def delete_product(request, product_id):
    """View to delete a product"""
    if request.method == 'POST':
        try:
            product = get_object_or_404(Product, id=product_id)
            product_name = product.product_name
            product.delete()
            messages.success(request, f'Product "{product_name}" deleted successfully!')
        except Exception as e:
            messages.error(request, f'Error deleting product: {str(e)}')
    return redirect('view_addproducts')

@login_required
def delete_product_image(request, image_id):
    """View to delete a specific product image"""
    if request.method == 'POST':
        try:
            image = get_object_or_404(ProductImage, id=image_id)
            product_id = image.product.id
            image.delete()
            messages.success(request, 'Product image deleted successfully!')
            return redirect('edit_productdet', product_id=product_id)
        except Exception as e:
            messages.error(request, f'Error deleting image: {str(e)}')
            return redirect('view_addproducts')
        
def products_view(request):
    products = Product.objects.prefetch_related('images').all()
    return render(request, 'products.html', {'products': products})

def product_detail_view(request, product_id):
    product = get_object_or_404(
        Product.objects.select_related('category').prefetch_related('images', 'descriptions'), 
        id=product_id
    )
    return render(request, 'product_detail.html', {'product': product})

@login_required
def cart_view(request):
    cart_items = CartItem.objects.filter(user=request.user).select_related('product')
    total_price = sum(item.get_total_price() for item in cart_items)
    
    context = {
        'cart_items': cart_items,
        'total_price': total_price,
    }
    return render(request, 'cart.html', context)

@require_POST
@login_required
def add_to_cart(request, product_id):
    try:
        quantity = int(request.POST.get('quantity', 1))
        product = get_object_or_404(Product, id=product_id)
        
        cart_item, created = CartItem.objects.get_or_create(
            user=request.user,
            product=product,
            defaults={'quantity': quantity}
        )
        
        if not created:
            cart_item.quantity += quantity
            cart_item.save()
        
        return redirect('cart')
    except Exception as e:
        print(f"Error adding to cart: {e}")  # For debugging
        return JsonResponse({'error': 'Failed to add to cart'}, status=400)

@require_POST
@login_required
def update_cart_quantity(request, item_id):
    cart_item = get_object_or_404(CartItem, id=item_id, user=request.user)
    quantity = int(request.POST.get('quantity', 1))
    
    # Ensure quantity is at least 1
    if quantity < 1:
        quantity = 1
    
    # Check if requested quantity exceeds stock
    if quantity > cart_item.product.stock:
        return JsonResponse({
            'error': 'Maximum stock reached',
            'max_stock_reached': True
        }, status=400)
    
    cart_item.quantity = quantity
    cart_item.save()
    
    return JsonResponse({
        'total_price': cart_item.get_total_price(),
        'cart_total': sum(item.get_total_price() for item in CartItem.objects.filter(user=request.user)),
        'max_stock_reached': quantity >= cart_item.product.stock
    })
    
@require_POST
@login_required
def remove_from_cart(request, item_id):
    cart_item = get_object_or_404(CartItem, id=item_id, user=request.user)
    cart_item.delete()
    return redirect('cart')

@login_required
def place_order(request):
    cart_items = CartItem.objects.filter(user=request.user)
    if cart_items.exists():
        # Add your order processing logic here
        cart_items.delete()  # Clear cart after order
        return JsonResponse({'success': True, 'message': 'Order placed successfully!'})
    return JsonResponse({'success': False, 'message': 'Cart is empty!'})

def toggle_wishlist(request, product_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Please login first'})
    
    product = get_object_or_404(Product, id=product_id)
    wishlist_item = Wishlist.objects.filter(user=request.user, product=product)
    
    if wishlist_item.exists():
        return JsonResponse({
            'status': 'error',
            'message': 'Product is already in your wishlist'
        })
    
    Wishlist.objects.create(user=request.user, product=product)
    return JsonResponse({
        'status': 'success',
        'message': 'Product added to wishlist'
    })

@require_POST
@login_required
def remove_from_wishlist(request, product_id):
    try:
        wishlist_item = Wishlist.objects.get(
            user=request.user,
            product_id=product_id
        )
        wishlist_item.delete()
        return JsonResponse({
            'status': 'success',
            'message': 'Product removed from wishlist'
        })
    except Exception as e:
        print(f"Error removing from wishlist: {e}")  # For debugging
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

def wishlist_view(request):
    wishlist_items = Wishlist.objects.filter(user=request.user).select_related('product')
    return render(request, 'wishlist.html', {'products': [item.product for item in wishlist_items]})