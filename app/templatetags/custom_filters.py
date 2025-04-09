from django import template
from datetime import timedelta

register = template.Library()

@register.filter(name='add_days')
def add_days(date, days):
    """Add the specified number of days to the date."""
    if date is None:
        return None
    try:
        return date + timedelta(days=int(days))
    except (ValueError, TypeError):
        return date 