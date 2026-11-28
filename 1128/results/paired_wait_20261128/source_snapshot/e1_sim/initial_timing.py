"""D-only timetable arithmetic in the same event order as physical execution.

Even Decimal addition is not associative at finite precision. Start from the
absolute clock and add each handling, flight and service duration separately;
do not translate a pre-summed relative duration into an absolute finish time.
"""
from .quantities import dec
from .constraints import distance

def load_end(cfg,orders,visit_order,start):
    t=dec(start)
    for oid in visit_order:
        t+=dec(cfg['drone_handling_setup_per_request'])+dec(cfg['drone_handling_per_kg'])*orders[oid]['weight']
    return t

def flight_end(cfg,network,orders,station,visit_order,takeoff):
    t=dec(takeoff);xy=network['stations'][station];service={}
    for oid in visit_order:
        t+=distance(xy,orders[oid]['xy'])/dec(cfg['drone_speed_km_min'])
        t+=dec(cfg['customer_service_per_request'])
        service[oid]=t;xy=orders[oid]['xy']
    t+=distance(xy,network['stations'][station])/dec(cfg['drone_speed_km_min'])
    return t,service
