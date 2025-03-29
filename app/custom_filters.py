from django import template

register = template.Library()

@register.filter
def sum_total_amount(orders):
    return sum(order.total_amount for order in orders)