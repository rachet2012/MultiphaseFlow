import math as m
import numpy as np
import scipy.constants as CONST
import scipy.optimize as sp
from unifloc_py.uniflocpy.uPVT.PVT_fluids import FluidStanding as PVT

class HasanKabirAnn():
    """
    Класс для расчета градиента давления в затрубном пространстве

    """
    def __init__(self, qu_gas_m3sec = 0.0005, qu_liq_m3sec: float = 0.001,
                rho_gas_kgm3: float = 0.679, rho_liq_kgm3: float = 860,
                sigma_Nm: float = 0.015, d_i_m: float = 0.05, d_o_m: float = 0.1 , 
                C0: float = 1.2, C1: float = 1.15, C2: float = 0.345, 
                mu_mix_pasec: float = 0.08, theta: float = 90) -> None:
        
        self.mu_gas_pasec = None
        self.mu_liq_pasec = None

        #добавить вязкости газа,жидкости
        self.qu_gas_m3sec = qu_gas_m3sec
        self.qu_liq_m3sec = qu_liq_m3sec
        self.rho_gas_kgm3 = rho_gas_kgm3
        self.rho_liq_kgm3 = rho_liq_kgm3
        self.sigma_Nm = sigma_Nm
        self.d_i_m = d_i_m
        self.d_o_m = d_o_m
        self.C0 = C0
        self.C1 = C1
        self.C2 = C2
        self.mu_mix_pasec = mu_mix_pasec
        self.theta = theta
        #C1 =1.2 в slug С1=1.15 в churn

        #рассчитанные
        self.f_m2 = None       
        self.d_equ_m = None
        self.v_d_msec = None
        self.vs_gas_msec = None
        self.vs_liq_msec = None
        self.v_mix_msec = None
        self.rho_m_rash_kgm3 = None #плотность смеси через расходную концентрацию газа
        self.number_Re = None


        self.vs_gas_bubble2slug_msec = None
        self.vs_gas_2annular_msec = None
        self.k_ratio_d = None
        self.friction_coeff = None
        self.Fca = None
        self.v_m_krit2disp_msec = None

        self.flow_pattern = None
        self.flow_pattern_name = None

        self.v_dt_msec = None

        self.epsi_s = None
        self.epsi_t = None

        self.epsi = None
        self.rho_mix_kgm3 = None

        self.calc_rash()
        self.calc_pattern()
        self.calc_rho_mix()

    def calc_rash(self):
        """
        ##Функция для расчета общих параметров потока

        """
        self.f_m2 =  CONST.pi * ((self.d_o_m/2)**2 - (self.d_i_m/2)**2)
        self.d_equ_m = self.d_o_m - self.d_i_m
        self.v_d_msec = 1.53 * (CONST.g * self.sigma_Nm * (self.rho_liq_kgm3 
                        - self.rho_gas_kgm3) / (self.rho_liq_kgm3)**2 ) ** 0.25
        self.vs_gas_msec = self.qu_gas_m3sec / self.f_m2
        self.vs_liq_msec = self.qu_liq_m3sec / self.f_m2
        self.v_mix_msec = (self.qu_gas_m3sec + self.qu_liq_m3sec) / self.f_m2
        self.rho_m_rash_kgm3 = self.qu_gas_m3sec / (self.qu_gas_m3sec + self.qu_liq_m3sec)
        self.number_Re = self.rho_m_rash_kgm3 * self.v_mix_msec * self.d_equ_m / self.mu_mix_pasec
        
             
    def _friction_coefficient_Gunn_Darling(self, initial_f):
        """
        ##Функция для определения коэффициента трения в турбулентном течении 
        Upward Vertical Two-Phase Flow Through an Annulus—Part I [15-27]

        """
        right_part = (4 * m.log(self.number_Re * (initial_f * (16 / self.Fca) **
                     (0.45 * m.exp(-(self.number_Re - 3000) / 10 ** 6))) ** 0.5) - 0.4)
        left_part = 1 / (initial_f * (16 / self.Fca) ** (0.45 * m.exp(-(self.number_Re - 
                    3000) / 10 ** 6))) ** 0.5

        return right_part - left_part   
        
    def _mixture_velocity_Caetano(self, initial_f):
        """
        ##Функция для определения критической скорости dispersed bubbly flow
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI

        """
        right_part = 2 * (initial_f ** 1.2) * (self.friction_coeff ** 0.4) * ((2 / self.d_equ_m) **
                     0.4) * ((self.rho_liq_kgm3 / self.sigma_Nm) ** 0.6 ) *(0.4 * self.sigma_Nm / (
                         (self.rho_liq_kgm3 - self.rho_gas_kgm3) * CONST.g) ** 0.5)

        left_part = 0.725 + 4.15 * (self.vs_gas_msec / initial_f) ** 0.5

        return right_part - left_part 

    def calc_pattern(self):
        """
        ##Функция для расчета критических значений скоростей
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        Upward Vertical Two-Phase Flow Through an Annulus—Part I for to dispersed

        """
        #bubble to slug transition [4]
        self.vs_gas_bubble2slug_msec = ((self.C0 * self.vs_liq_msec + self.v_d_msec) / (4 
                                    - self.C0)) * np.sin(self.theta * np.pi/180)

        #to annular transirion [17]
        self.vs_gas_2annular_msec = 3.1 * (self.sigma_Nm * CONST.g * (self.rho_liq_kgm3 - 
                                    self.rho_gas_kgm3) / (self.rho_gas_kgm3) ** 2) ** 0.25

        #bubble/slug to dispersed transition [6]
        #Upward Vertical Two-Phase Flow Through an Annulus—Part I [15-27]
        self.k_ratio_d = self.d_o_m / self.d_i_m
        self.Fca = (16 * (1 - self.k_ratio_d) ** 2 /
                    ((1 - self.k_ratio_d ** 4) / (1 - self.k_ratio_d ** 2) -
                     (1 - self.k_ratio_d ** 2) / m.log(1 / self.k_ratio_d)))
        if self.number_Re < 3000:  # laminar flow
            self.friction_coeff = self.Fca / self.number_Re
        else:  # turbulent flow
            self.friction_coeff = float(sp.fsolve(self._friction_coefficient_Gunn_Darling, 0.000005))
        self.v_m_krit2disp_msec = float(sp.fsolve(self._mixture_velocity_Caetano, 1000))

        self.set_flow_pattrn()

    def set_flow_pattrn(self):
        """
        ## Функция определения структуры потока

        """
        if self.vs_gas_msec >= self.vs_gas_2annular_msec:
            self.flow_pattern = 4
            self.flow_pattern_name = 'Annular flow pattern - Кольцевой режим'
        elif self.vs_gas_msec >= self.vs_gas_bubble2slug_msec and (0.25 * self.vs_gas_msec) < 0.52 and self.v_mix_msec < self.v_m_krit2disp_msec:
            self.flow_pattern = 2
            self.flow_pattern_name = 'Slug flow pattern - Пробковый режим'
        elif self.vs_gas_msec >= self.vs_gas_bubble2slug_msec and (0.25 * self.vs_gas_msec) >= 0.52:
            self.flow_pattern = 3
            self.flow_pattern_name = 'Chug flow pattern - Вспененный режим'
        elif self.vs_gas_msec <= self.vs_gas_bubble2slug_msec and self.v_mix_msec < self.v_m_krit2disp_msec:
            self.flow_pattern = 0
            self.flow_pattern_name = 'Bubble flow pattern - пузырьковый режим'
        elif self.v_mix_msec >= self.v_m_krit2disp_msec:
            self.flow_pattern = 1
            self.flow_pattern_name = 'Dispersed bubble flow pattern - дисперсионно-пузырьковый режим'       

    def calc_bubbly(self) -> float:
        """
        ##Функция для расчета истинной объемной концентрации газа в bubbly flow

        """
        self.v_d_msec = (1.53 * (CONST.g * self.sigma_Nm * (self.rho_liq_kgm3 - self.rho_gas_kgm3)
                     / (self.rho_liq_kgm3)**2 ) ** 0.25) #3

        self.v_gas_msec = self.C0 * self.v_mix_msec + self.v_d_msec #1
        self.epsi = self.vs_gas_msec / self.v_gas_msec #2


    def calc_slug_churn(self, C) -> float:
        """
        ##Функция для расчета истинной объемной концентрации газа в slug и churn flow

        """
        self.v_d_msec = (1.53 * (CONST.g * self.sigma_Nm * (self.rho_liq_kgm3 - self.rho_gas_kgm3) 
                        / (self.rho_liq_kgm3)**2 ) ** 0.25) #3

        self.v_dt_msec = (0.345 + 0.1 * (self.d_i_m / self.d_o_m)) *((CONST.g * self.d_o_m * (
                        self.rho_liq_kgm3 - self.rho_gas_kgm3)/(self.rho_gas_kgm3)) ** 0.5) #14

        self.epsi_s = self.vs_gas_msec / (self.C0 * self.v_mix_msec + self.v_dt_msec)#расчет rho_s в градиенте

        self.epsi_t = self.vs_gas_msec / (C * self.v_mix_msec + self.v_dt_msec) #7

        if self.vs_gas_msec > 0.4:
            self.len_s_m = 0.1 * (C * self.v_mix_msec + self.v_d_msec) / self.vs_gas_msec #10b
            
            self.epsi = (1 - self.len_s_m) * self.epsi_t + 0.1  #9a

        else:
            self.len_s_m = 0.25 * (C * self.v_mix_msec + self.v_d_msec) #11
            self.epsi = (1 - self.len_s_m) * self.epsi_t + 0.25 * self.vs_gas_msec #9b

    def _actual_film_length(self, initial_llf):
        """
        Функция для вычисления фактический длины пленки жидкости в slug/churn
        Уравнение  [44]
        """
        coef_b = (-2 * (1 - self.vs_gas_msec / self.v_dt_msec) * ((self.vs_gas_msec - self.v_gls #45
                    *(1 - self.h_ls) / self.v_dt_msec)) * self.len_ls + (2 / CONST.g) * (self.v_dt_msec 
                    - self.v_lls) ** 2 * self.h_ls **2) / (1 - self.vs_gas_msec / self.v_dt_msec) ** 2

        coef_c = (((self.vs_gas_msec - self.v_gls * (1 - self.h_ls)) / self.v_dt_msec * self.len_ls)
                        / (1 - self.vs_gas_msec / self.v_dt_msec)) ** 2
        return initial_llf ** 2 - coef_b * initial_llf + coef_c

    def _acceler_grad_p(self) :
        """
        Функция для нахождения градиента давления на ускорения в churn и slug flow
        [20,23,38,44,47] H_ls =0.80(но можно так же найти как 1-epsi_s)? В методиках по разному

        """
        self.h_ls = 1 - self.epsi_s
        self.h_lf = 1 - self.epsi_t
        self.len_ls = 16 * self.d_equ_m #38
        self.len_su = self.len_ls / self.len_s_m
        self.v_lls = ((self.vs_liq_msec + self.vs_gas_msec) - 1.53 * ((self.rho_liq_kgm3 - self.rho_gas_kgm3) #23
                    * CONST.g * self.sigma_Nm / (self.rho_liq_kgm3) ** 2) ** 0.25 * self.h_ls ** 0.5 * (1 - self.h_ls)) 
        self.v_gls = (1.53 * ((self.rho_liq_kgm3 - self.rho_gas_kgm3) * CONST.g * self.theta / (self.rho_liq_kgm3) #20
                        ** 2) ** 0.25 * self.h_ls ** 0.5) - self.v_lls
        self.act_len_lf = float(sp.fsolve(self._actual_film_length, 0.05))
        self.v_llf = (CONST.g * 2 * self.act_len_lf) ** 0.5 - self.v_dt_msec #47
        grad_p_acc = self.rho_liq_kgm3 * (self.h_lf / self.len_su) * (self.v_llf - self.v_dt_msec) * (self.v_llf - self.v_lls)
        return grad_p_acc

    def _ratio_t(self):
        """
        Функция для вычисления отношения толщины пленок жидкости в кольцевом потоке [87,88]
        
        """
        angle_wt_average = (1 / (1 - self.k_ratio_d ** 2) * (2 * m.asin(self.k_ratio_d) + 2 * self.k_ratio_d #[88] арксинус посмотреть в чем возвращает
                             * (1 - self.k_ratio_d ** 2) ** 0.5 - CONST.pi * self.k_ratio_d ** 2))
        t_ratio = angle_wt_average / ((2 * CONST.pi - angle_wt_average) * self.k_ratio_d)
        return t_ratio

    def calc_hl_total_annular(self):
        """
        Функция для вычисления концентрации жидкости в потоке кольцевой структуры[77]
        Допустил, что толщина пленки жидкости на внешней(или внутренней) трубе известна()
        """
        self.delta_o = 0.005
        self.delta_i = self.delta_o * self._ratio_t()
        self.phi = (10 ** 4 * self.vs_gas_msec * self.mu_gas_pasec / self.sigma_Nm * (self.rho_gas_kgm3 #79
                    / self.rho_liq_kgm3) ** 0.5)
        self.fe = 1 - m.exp((self.phi - 1.5) * (-0.125))
        self.hl_total = (4 / ((self.d_o_m) * (1 - self.k_ratio_d ** 2)) * (self.delta_o * (1 - self.delta_o / self.d_o_m) 
                        + self.delta_i * self.k_ratio_d * (1 + self.delta_i / self.d_i_m) + self.vs_liq_msec * self.fe 
                        / ((self.vs_liq_msec * self.fe + self.vs_gas_msec) * (1 - self.k_ratio_d ** 2)) * 
                        (1 - self.k_ratio_d ** 2 - 4 * self.delta_o / self.d_o_m * (1 - self.delta_o / self.d_o_m) 
                        - 4 *self.delta_i *self.k_ratio_d / self.d_o_m * (1 + self.delta_i / self.d_i_m))))
        self.epsi = 1 - self.hl_total
        
        

    def calc_rho_mix(self):
        """
        ##Функция для расчета плотности смеси

        """
        if self.flow_pattern == 0 or self.flow_pattern == 1:
            self.calc_bubbly()
        elif self.flow_pattern == 2:
            self.calc_slug_churn(self.C0)
        elif self.flow_pattern == 3: #tut3
            self.calc_slug_churn(self.C1)
        elif self.flow_pattern == 4:
            self.cals_hl_total_aanular()

            pass
        
        self.rho_mix_kgm3 = self.rho_liq_kgm3 * (1 - self.epsi) + self.rho_gas_kgm3 * self.epsi


   


    def calc_pressure_gradient(self):
        """
        ##Функция для расчета градиента давления
        Upward Vertical Two-Phase Flow Through an Annulus—Part II
        
        """
        if self.flow_pattern == 0 or self.flow_pattern == 1: #[5-14]
            self.density_grad_pam = self.rho_mix_kgm3 * CONST.g * np.sin(self.theta * np.pi/180)

            self.friction_grad_pam = (4 * self.friction_coeff / (self.d_o_m - self.d_i_m) 
                                     * self.v_mix_msec ** 2 / 2)

            self.acceleration_grad_pam = 0

        elif self.flow_pattern == 2 or self.flow_pattern == 3: #предположил что для slug и churn одна методика. Концентрацию воды нашел как 1 - epsi
            
            self.rho_slug_kgm3 = self.rho_gas_kgm3 * self.epsi_s + self.rho_liq_kgm3 * (1-self.epsi_t) # В соответствии c [51] 

            self.density_grad_pam = self.rho_slug_kgm3 * CONST.g * self.len_s_m #[50]

            self.friction_grad_pam = ((2 * self.friction_coeff / self.d_equ_m * self.rho_slug_kgm3) #[53]
                                     * (self.vs_gas_msec + self.vs_liq_msec) **2 * self.len_s_m)

            self.acceleration_grad_pam = self._acceler_grad_p () 

        elif self.flow_pattern == 4:

            self.density_grad_pam = 0

            self.friction_grad_pam = 0

            self.acceleration_grad_pam = 0

        self.result_grad_pam = self.friction_grad_pam  + self.density_grad_pam + self.acceleration_grad_pam

        return self.result_grad_pam


        

if __name__ == '__main__':

    check = HasanKabirAnn()
    check.calc_pressure_gradient()
    print(check.__dict__)

