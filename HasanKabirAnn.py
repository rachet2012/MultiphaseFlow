import math as m
import numpy as np
from numpy.core.fromnumeric import put
import scipy.constants as CONST
import scipy.optimize as sp
from scipy.integrate import solve_ivp
from unifloc.pvt.fluid_flow import FluidFlow
import unifloc.pipe._friction as fr
from math import fabs
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

class HasanKabirAnn():
    """
    Класс для расчета градиента давления в затрубном пространстве
    Определяются структура потока, истинная концентрация газа, плотность смеси,
    градиенты на гравитацию, трение, ускорение.
    Вычисляется распеределение давление в затрубном пространстве.
    """

    def __init__(self, fluid ,  d_i_m: float = 73, d_o_m: float = 142,
         theta: float = 90, abseps:float = 2.54) -> None:
        """
        :param fluid: PVT модель флюида
        :param d_i_m: внешний диаметр НКТ, мм
        :param d_o_m: внутренний диаметр ЭК, мм
        :param theta: угол наклона скважины
        :param abseps: абсолютная шероховатость стенок трубы, м
        """
        self.fluid = fluid

        self.abseps =abseps / 100000
        self.d_i_m = d_i_m / 1000
        self.d_o_m = d_o_m / 1000
        self.theta = theta
        

    def calc_par(self):
        f_m2 =  CONST.pi * ((self.d_o_m/2)**2 - (self.d_i_m/2)**2)
        self.d_equ_m = self.d_o_m - self.d_i_m
        self.vs_gas_msec = self.fluid.qg / f_m2
        self.vs_liq_msec = self.fluid.ql / f_m2
        self.v_mix_msec = (self.fluid.qg + self.fluid.ql) / f_m2
        self.k_ratio_d = self.d_i_m / self.d_o_m


    def _mixture_velocity_Caetano(self, initial_f):
        """
        Метод для определения критической скорости dispersed bubbly flow
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        """
        rho_m_rash_kgm3 = (self.fluid.qg / (self.fluid.qg + self.fluid.ql) 
                                * self.fluid.rg + (1 - self.fluid.qg / (self.fluid.qg + 
                                self.fluid.ql)) * self.fluid.rl)
        friction_coeff = self._friction_coefv2(rho_m_rash_kgm3)
        right_part = 2 * (initial_f ** 1.2) * (friction_coeff** 0.4) * ((2 / self.d_equ_m) **
                     0.4) * ((self.fluid.rl/ self.fluid.stlg) ** 0.6 ) *(0.4 * self.fluid.stlg / (
                         (self.fluid.rl- self.fluid.rg) * CONST.g) ** 0.5)
        left_part = 0.725 + 4.15 * (self.vs_gas_msec / initial_f) ** 0.5
        return right_part - left_part
    
    def _friction_coefficient_Gunn_Darling(self, initial_ff):
        """
        Метод для определения коэффициента трения в турбулентном течении 
        Upward Vertical Two-Phase Flow Through an Annulus—Part I [15-27]
        """
        right_part = (4 * m.log(self.num_Re* (initial_ff * (16 / self.Fca) **
                     (0.45 * m.exp(-(self.num_Re - 3000) / (10 ** 6)))) ** 0.5) - 0.4)
        left_part = 1 / (initial_ff * (16 / self.Fca) ** (0.45 * m.exp(-(self.num_Re - 
                    3000) / 10 ** 6))) ** 0.5
        return right_part - left_part  


    def _friction_coefv2(self, rho):
        """
        Метод для определения коэффициента трения
        :param rho: фактическая плотность ГЖС
        :return: коэффициент трения
        """
        frict = fr.Friction(self.d_o_m)
        eps = self.abseps / self.d_o_m
        mu_mix_pasec = (self.vs_liq_msec / self.v_mix_msec * self.fluid.mul 
                            + self.vs_gas_msec / self.v_mix_msec * self.fluid.mug)
        number_Re_s = frict.calc_n_re(rho, self.v_mix_msec, mu_mix_pasec)
        ff = frict.calc_norm_ff(number_Re_s, eps, 1)
        return ff

    def _calc_pattern(self):
        """
        Метод для расчета критических значений скоростей
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        Upward Vertical Two-Phase Flow Through an Annulus—Part I for to dispersed
        """
        #bubble to slug transition [4]
        v_d_msec = 1.53 * (CONST.g * self.fluid.stlg * (self.fluid.rl
                        - self.fluid.rg) / (self.fluid.rl)**2 ) ** 0.25
        self.vs_gas_bubble2slug_msec = ((1.2 * self.vs_liq_msec + v_d_msec) / (4 
                                    - 1.2)) * np.sin(self.theta * np.pi/180)
        #to annular transirion [17]
        self.vs_gas_2annular_msec = (3.1 * (self.fluid.stlg * CONST.g * (self.fluid.rl- 
                                    self.fluid.rg) / (self.fluid.rg) ** 2) ** 0.25 + 1)
        #bubble/slug to dispersed transition [6]
        try:
            self.v_m_krit2disp_msec = fabs(float(sp.fsolve(self._mixture_velocity_Caetano, 6, maxfev=20)))
        except:
            self.v_m_krit2disp_msec = 10

        self._set_flow_pattrn()

    def _actual_friction_coef(self, rho):
        """
        Метод для расчета коэффициента трения
        :param rho: плотность смеси по различным корреляциям, кг/м3
        """
        frict = fr.Friction(self.d_o_m)
        mu_mix_pasec = (self.vs_liq_msec / self.v_mix_msec * self.fluid.mul 
                            + self.vs_gas_msec / self.v_mix_msec * self.fluid.mug)
        self.num_Re = frict.calc_n_re(rho, self.v_mix_msec, mu_mix_pasec)
        self.Fca = (16 * (1 - self.k_ratio_d) ** 2 /
                    ((1 - self.k_ratio_d ** 4) / (1 - self.k_ratio_d ** 2) -
                     (1 - self.k_ratio_d ** 2) / m.log(1 / self.k_ratio_d)))
        if self.num_Re < 3000:  # laminar flow
            fric = self.Fca / self.num_Re
        else:  # turbulent flow
            fric = float(sp.fsolve(self._friction_coefficient_Gunn_Darling, 0.02))
        return fric

    def _set_flow_pattrn(self):
        """
        Метод для определения структуры потока
        """
        if self.vs_gas_msec >= self.vs_gas_2annular_msec and 1000 > 10000:
            self.flow_pattern = 4
            self.flow_pattern_name = 'Annular flow pattern - Кольцевой режим'
        elif self.vs_gas_msec <= self.vs_gas_bubble2slug_msec and self.v_mix_msec < self.v_m_krit2disp_msec:
            self.flow_pattern = 0
            self.flow_pattern_name = 'Bubble flow pattern - пузырьковый режим'
        elif self.vs_gas_msec >= self.vs_gas_bubble2slug_msec and (0.25 * self.vs_gas_msec) < 0.52 and self.v_mix_msec < self.v_m_krit2disp_msec:
            self.flow_pattern = 2
            self.flow_pattern_name = 'Slug flow pattern - Пробковый режим'
        elif self.vs_gas_msec >= self.vs_gas_bubble2slug_msec and (0.25 * self.vs_gas_msec) >= 0.52:
            self.flow_pattern = 3
            self.flow_pattern_name = 'Chug flow pattern - Вспененный режим'
        elif self.v_mix_msec >= self.v_m_krit2disp_msec:
            self.flow_pattern = 1
            self.flow_pattern_name = 'Dispersed bubble flow pattern - дисперсионно-пузырьковый режим'       

    def _calc_bubbly(self) -> float:
        """
        Метод для расчета истинной объемной концентрации газа в bubbly flow
        """
        v_d_msec = (1.53 * (CONST.g * self.fluid.stlg * (self.fluid.rl- self.fluid.rg)
                     / (self.fluid.rl)**2 ) ** 0.25) #3
        v_gas_msec = 1.2 * self.v_mix_msec + v_d_msec #1
        self.epsi = self.vs_gas_msec / v_gas_msec #2

    def _calc_slug_churn(self) -> float:
        """
        Метод для расчета истинной объемной концентрации газа в slug и churn flow
        """
        self.v_dt_msec = (1.2 *(self.vs_gas_msec + self.vs_liq_msec) + 0.345 * 
                        (CONST.g * (self.d_i_m + self.d_o_m)) ** 0.5 
                        * np.sin(self.theta * np.pi/180) ** 0.5 
                        * (1 + np.cos(self.theta * np.pi/180)) ** 1.2)#17

        self.epsi_s = self.vs_gas_msec / (1.2 * self.v_mix_msec + self.v_dt_msec)
        self.epsi_t = self.vs_gas_msec / (1.15 * self.v_mix_msec + self.v_dt_msec) #7

        if self.vs_gas_msec > 0.4:
            self.len_s_m = 0.1 / self.epsi_s 
            self.epsi = (1 - self.len_s_m) * self.epsi_t + 0.1  #9a
        else:
            self.len_s_m = 0.25 * self.vs_gas_msec / self.epsi_s
            self.epsi = (1 - self.len_s_m) * self.epsi_t + 0.25 * self.vs_gas_msec #9b

    def _actual_film_length(self):
        """
        Метод для вычисления фактический длины пленки жидкости в slug/churn
        Уравнение  [44]
        
        """
        coef_b = (-2 * (1 - self.vs_gas_msec / self.v_dt_msec) * ((self.vs_gas_msec - self.v_gls #45
                    *(1 - self.h_ls)) / self.v_dt_msec) * self.len_ls + (2 / CONST.g) * (self.v_dt_msec 
                    - self.v_lls) ** 2 * self.h_ls **2) / (1 - self.vs_gas_msec / self.v_dt_msec) ** 2

        coef_c = (((self.vs_gas_msec - self.v_gls * (1 - self.h_ls)) / self.v_dt_msec * self.len_ls)
                        / (1 - self.vs_gas_msec / self.v_dt_msec)) ** 2
        
        discr = (coef_b ** 2 - 4 * coef_c)
        #дискриминант отрицательный, уравнение в каких то диапазонах не решается
        if discr > 0 :
            x1 = (-coef_b + m.sqrt(discr)) / 2 
            x2 = (-coef_b - m.sqrt(discr)) / 2 
            if x1 >= 0 and x2 < 0:
                self.resh = x1
            elif x2 >= 0 and x1 <0:
                self.resh = x2
            elif x1 < 0 and x2 < 0:
                self.resh = 0.000001 
            elif x1 > x2:
                self.resh = 0.000001 
            elif x1 < x2:
                self.resh = 0.000001 
        elif discr == 0:
            self.resh = -coef_b / 2
        else:
            self.resh = 0.000001 
        return self.resh

    def _acceler_grad_p(self) :
        """
        Метод для нахождения градиента давления на ускорения в churn и slug flow
        [20,23,38,44,47] 
        :return: градиент давления на ускорение
        """
        self.h_ls = 1 - self.epsi_s
        h_lf = 1 - self.epsi_t
        self.len_ls = 16 * self.d_equ_m #38
        len_su = self.len_ls / self.len_s_m
        self.v_lls = ((self.vs_liq_msec + self.vs_gas_msec) - 1.53 * ((self.fluid.rl- self.fluid.rg) #23
                    * CONST.g * self.fluid.stlg / (self.fluid.rl**2)) ** 0.25 * self.h_ls ** 0.5 * (1 - self.h_ls)) 
        self.v_gls = ((1.53 * ((self.fluid.rl- self.fluid.rg) * CONST.g * self.fluid.stlg / (self.fluid.rl**2) #20
                        ) ** 0.25 * self.h_ls ** 0.5) + self.v_lls)

        act_len_lf = (self._actual_film_length())
        v_llf = fabs((CONST.g * 2 * act_len_lf) ** 0.5 - self.v_dt_msec) #47
        grad_p_acc = (self.fluid.rl* (h_lf / len_su) * (v_llf - self.v_dt_msec) 
                    * (v_llf - self.v_lls))
        if grad_p_acc < 0:
            grad_p_acc = 0
        else:
            grad_p_acc = grad_p_acc
        return grad_p_acc

    def _acceler_grad_p_annular(self):
        """
        Метод для расчета потерь на ускорения в кольцевом режиме потока
        """
        self.v_dt_msec = (0.345 + 0.1 * (self.d_i_m / self.d_o_m)) *((CONST.g * self.d_o_m * (
                        self.fluid.rl- self.fluid.rg)/(self.fluid.rg)) ** 0.5)
        len_su = 1
        act_len_lf = len_su
        v_llf = (CONST.g * 2 * act_len_lf) ** 0.5 - self.v_dt_msec
        grad_p_acc_an = (self.fluid.rl* (self.hl_total / len_su) * (v_llf - self.v_dt_msec) 
                    * v_llf)
        return grad_p_acc_an

    def _ratio_t(self):
        """
        Метод для вычисления отношения толщины пленок жидкости в кольцевом потоке [87,88]
        :return:
        
        """
        angle_wt_average = (1 / (1 - self.k_ratio_d ** 2) * (2 * m.asin(self.k_ratio_d) + 2 * #[88]
                             self.k_ratio_d  * (1 - self.k_ratio_d ** 2) ** 0.5 - CONST.pi * 
                            self.k_ratio_d ** 2))
        t_ratio = angle_wt_average / ((2 * CONST.pi - angle_wt_average) * self.k_ratio_d)
        return t_ratio

    def _calc_hl_total_annular(self):
        """
        Метод для вычисления концентрации газа в потоке кольцевой структуры[77]
        Допустил, что толщина пленки жидкости на внешней(или внутренней) трубе известна()
        """
        delta_o = 0.005
        delta_i = delta_o * self._ratio_t()
        phi = (10 ** 4 * self.vs_gas_msec * self.fluid.mug / self.fluid.stlg * (self.fluid.rg #[79]
                    / self.fluid.rl) ** 0.5)
        fe = 1 - m.exp((phi - 1.5) * (-0.125))
        self.hl_total = (4 / ((self.d_o_m) * (1 - self.k_ratio_d ** 2)) * (delta_o * (1 - delta_o / self.d_o_m) #[77]
                        + delta_i * self.k_ratio_d * (1 + delta_i / self.d_i_m) + self.vs_liq_msec * fe 
                        / ((self.vs_liq_msec * fe + self.vs_gas_msec) * (1 - self.k_ratio_d ** 2)) * 
                        (1 - self.k_ratio_d ** 2 - 4 * delta_o / self.d_o_m * (1 - delta_o / self.d_o_m) 
                        - 4 * delta_i *self.k_ratio_d / self.d_o_m * (1 + delta_i / self.d_i_m))))
        self.epsi = 1 - self.hl_total 
          
    def calc_rho_mix(self):
        """
        Метод для расчета плотности смеси
        """
        self._calc_pattern()
        if self.flow_pattern == 0 or self.flow_pattern == 1:
            self._calc_bubbly()
        elif self.flow_pattern == 2:
            self._calc_slug_churn()
        elif self.flow_pattern == 3: 
            self._calc_slug_churn()
        elif self.flow_pattern == 4:
            self._calc_hl_total_annular()
        self.rho_mix_kgm3 = self.fluid.rl* (1 - self.epsi) + self.fluid.rg * self.epsi

    def fanning_f(self,rho) :
        frict = fr.Friction(self.d_o_m)
        mu_mix_pasec = (self.vs_liq_msec / self.v_mix_msec * self.fluid.mul 
                            + self.vs_gas_msec / self.v_mix_msec * self.fluid.mug)
        Re = frict.calc_n_re(rho, self.v_mix_msec, mu_mix_pasec)
        abs_rough = self.abseps * 3.281
        d = self.d_equ_m * 3.281
        f_lam = (16 * (1 - self.k_ratio_d) ** 2 /
                ((1 - self.k_ratio_d ** 4) / (1 - self.k_ratio_d ** 2) -
                (1 - self.k_ratio_d ** 2) / m.log(1 / self.k_ratio_d)))
        inv_sqrt_f = -4 * np.log10(0.2698 * (abs_rough / d) - (5.0452 / Re) * np.log10(.3539 * (abs_rough / d)**1.1098 + 5.8506 / Re**.8981))
        f_turb = (1 / inv_sqrt_f)**2
    
        if(Re <= 2000):
            fri = f_lam
        elif((Re > 2000) & (Re < 4000)):
            fri = (f_lam * (4000 - Re) + f_turb * (Re - 2000)) / 2000
        elif(Re >= 4000):
            fri = f_turb
        return fri

    def calc_pressure_gradient(self):
        """
        Метод для расчета градиента давления
        Upward Vertical Two-Phase Flow Through an Annulus—Part II
        :param p: текущее давление, Па 
        :param t: текущая температура, К
        :return: суммарный градиент давления
        
        """
        self.calc_par()
        self.calc_rho_mix()  

        if self.flow_pattern == 0 or self.flow_pattern == 1: #[5-14]
            self.density_grad_pam = self.rho_mix_kgm3 * CONST.g * np.sin(self.theta * np.pi/180)

            # friction_coeff_s = self._friction_coefv2(self.rho_mix_kgm3)
            # friction_coeff_s = self.fanning_f(self.rho_mix_kgm3)
            friction_coeff_s = self._actual_friction_coef(self.rho_mix_kgm3)
            self.friction_grad_pam = ((4 * friction_coeff_s / (self.d_equ_m ) 
                                     * self.v_mix_msec ** 2 / 2) * self.rho_mix_kgm3)

            self.acceleration_grad_pam = 0
            
        elif self.flow_pattern == 2 or self.flow_pattern == 3 : 
            self.rho_slug_kgm3 = self.fluid.rg * self.epsi_s + self.fluid.rl* (1 - self.epsi_s) # В соответствии c [51] 

            self.density_grad_pam = self.rho_slug_kgm3 * CONST.g  * self.len_s_m *  np.sin(self.theta * np.pi/180) #[50]

            # friction_coeff_s =  self._friction_coefv2(self.rho_slug_kgm3)
            # friction_coeff_s = self.fanning_f(self.rho_slug_kgm3)
            friction_coeff_s = self._actual_friction_coef(self.rho_slug_kgm3)
            self.friction_grad_pam = ((2 * friction_coeff_s / self.d_equ_m * self.rho_slug_kgm3) #[53]
                                     * (self.vs_gas_msec + self.vs_liq_msec) **2 * self.len_s_m)

            self.acceleration_grad_pam = self._acceler_grad_p() 

        elif self.flow_pattern == 4:# над ускорением подумать
            self.density_grad_pam = self.rho_mix_kgm3 * CONST.g * np.sin(self.theta * np.pi/180)

            friction_coeff_s = self._actual_friction_coef(self.rho_mix_kgm3)
            self.friction_grad_pam = (4 * friction_coeff_s / (self.d_o_m - self.d_i_m) 
                                     * self.v_mix_msec ** 2 / 2) * self.rho_mix_kgm3

            self.acceleration_grad_pam = self._acceler_grad_p_annular()


        self.result_grad_pam = self.friction_grad_pam  + self.density_grad_pam + self.acceleration_grad_pam

        return self.result_grad_pam



if __name__ == '__main__':

    def grad_func(h, pt, d_i, d_o, rp, ql, wct):
        pvt_model =  {"black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
                                        "rp": rp,
                                        "oil_correlations":
                                        {"pb": "Standing", "rs": "Standing",
                                        "rho": "Standing","b": "Standing",
                                        "mu": "Beggs", "compr": "Vasquez"},
                            "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
                                                "z": "Dranchuk", "mu": "Lee"},
                            "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                    "rho": "Standing", "mu": "McCain"}}}
        PVT = FluidFlow(ql/86400, wct, pvt_model)
        PVT.calc_flow(pt[0],pt[1])
        test3 = HasanKabirAnn(d_i_m = d_i, d_o_m = d_o, fluid = PVT)
        dp_dl = test3.calc_pressure_gradient() 
        dt_dl = 0.03
        return dp_dl, dt_dl

    def func_p_list(p_head, t_head, h, d_i, d_o, rp, qu_liq, wct):
        p0,t0 = p_head, t_head
        h0 = 0
        h1 = h
        steps = [i for i in range(h0, h1+50, 50)]
        sol = solve_ivp(grad_func, 
            t_span=(h0, h1), 
            y0=[p0, t0], 
            t_eval=steps,
            max_step = 55,
            args=(d_i, 
            d_o,
            rp,
            qu_liq, 
            wct,)
        ) 
        return sol.y, 
    
    for i in range(0, 400,10):
        tt = func_p_list(p_head = 15, t_head=293, h=2400, d_i = 73, d_o=142, rp=i, qu_liq=600, wct=0.4)
        vr1 = tt[0]
        vr2 = vr1[0]
        vr3= vr2[-1]
        print(vr3/101325)

 
      