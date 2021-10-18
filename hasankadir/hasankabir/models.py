from django.db import models

# Create your models here.
#        
class Database(models.Model):
    """
    :param qu_gas_m3day: дебит скважины по газу, м3/сут
    :param qu_liq_m3day: дебит скважины по жидкости, м3/сут
    :param d_i_m: внешний диаметр НКТ, мм
    :param d_o_m: внутренний диаметр ЭК, мм
    :param h: глубина скважины, м 
    :param p_head: давление на устье скважины, атм
    :param t_head: температура на устье скважины, С
    :param wct: обводненность продукции, дол.ед   
    """
    
    qu_gas_m3day = models.FloatField()  
    qu_liq_m3day = models.FloatField()  
    d_i_m = models.FloatField()  
    d_o_m = models.FloatField()  
    h = models.FloatField()  
    p_head = models.FloatField()  
    t_head = models.FloatField()  
    wct = models.FloatField()  
    results = models.CharField()