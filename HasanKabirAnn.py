import math as m
import numpy as np
import scipy.constants as CONST
import scipy.optimize as sp
from scipy.integrate import solve_ivp
from unifloc.pvt.fluid_flow import FluidFlow
import unifloc.pipe._friction as fr

#Новый коэфициент трения, ошибка меньше. Старый убрать

class HasanKabirAnn(FluidFlow):
    """
    Класс для расчета градиента давления в затрубном пространстве
    Определяются структура потока, истинная концентрация газа, плотность смеси,
    градиенты на гравитацию, трение, ускорение.
    Вычисляется распеределение давление в затрубном пространстве.
    """
    def __init__(self, qu_gas_m3day:float = 10, qu_liq_m3day: float = 432, d_i_m: float = 5,
                 d_o_m: float = 10,theta: float = 90, h:float = 2000, p_head:float = 5, 
                 t_head:float = 20, wct:float = 1, abseps:float = 0.1) -> None:
        """
        :param qu_gas_m3day: дебит скважины по газу, м3/сут
        :param qu_liq_m3day: дебит скважины по жидкости, м3/сут
        :param rho_gas_kgm3: плотность газа, кг/м3
        :param d_i_m: внешний диаметр НКТ, мм
        :param d_o_m: внутренний диаметр ЭК, мм
        :param theta: угол наклона скважины
        :param h: глубина скважины, м 
        :param p_head: давление на устье скважины, атм
        :param t_head: температура на устье скважины, С
        :param wct: обводненность продукции, дол.ед
        :param abseps: абсолютная шероховатость стенок трубы, м
        :param rp: газовый фактор, м3/м3

        """
        self.qu_gas_m3sec = qu_gas_m3day / 86400
        self.qu_liq_m3sec = qu_liq_m3day / 86400
        self.rp = None
        self.wct = wct

        self.p_head = p_head * (10 ** 5)
        self.t_head = t_head + 273

        self.mu_gas_pasec = None
        self.mu_liq_pasec = None
        self.rho_gas_kgm31 = None
        self.rho_liq_kgm3 = None
        self.sigma_Nm = None

        self.abseps =abseps / 100
        self.d_i_m = d_i_m / 100
        self.d_o_m = d_o_m / 100
        self.h = h
        self.theta = theta

        self.C0 = 1.2
        self.C1 = 1.15
        self.C2 = 0.345

        #рассчитанные
        self.f_m2 = None       
        self.d_equ_m = None
        self.v_d_msec = None
        self.vs_gas_msec = None
        self.vs_liq_msec = None
        self.v_mix_msec = None
        self.rho_m_rash_kgm3 = None #плотность смеси через расходную концентрацию газа
        self.number_Re = None
        self.mu_mix_pasec = None

        self.vs_gas_bubble2slug_msec = None
        self.vs_gas_2annular_msec = None



        self.eps = None
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

        self.k_ratio_d = None

        self.grad_t = 0.03
        
        self.calc_PVT(self.p_head, self.t_head)
        self.calc_rash()
        self.calc_pattern()
        self.calc_rho_mix()

    def calc_rash(self):
        """
        Метод для расчета общих параметров потока
        """
        self.f_m2 =  CONST.pi * ((self.d_o_m/2)**2 - (self.d_i_m/2)**2)
        self.d_equ_m = self.d_o_m - self.d_i_m
        self.v_d_msec = 1.53 * (CONST.g * self.sigma_Nm * (self.rho_liq_kgm3 
                        - self.rho_gas_kgm31) / (self.rho_liq_kgm3)**2 ) ** 0.25
        self.vs_gas_msec = self.qu_gas_m3sec / self.f_m2
        self.vs_liq_msec = self.qu_liq_m3sec / self.f_m2
        self.v_mix_msec = (self.qu_gas_m3sec + self.qu_liq_m3sec) / self.f_m2
        self.rho_m_rash_kgm3 = (self.qu_gas_m3sec / (self.qu_gas_m3sec + self.qu_liq_m3sec) 
                                * self.rho_gas_kgm31 + (1 - self.qu_gas_m3sec / (self.qu_gas_m3sec + 
                                self.qu_liq_m3sec)) * self.rho_liq_kgm3)
        self.mu_mix_pasec = (self.vs_liq_msec / self.v_mix_msec * self.mu_liq_pasec 
                            + self.vs_gas_msec / self.v_mix_msec * self.mu_gas_pasec)
        self.number_Re = self.rho_m_rash_kgm3 * self.v_mix_msec * self.d_equ_m / self.mu_mix_pasec
        self.k_ratio_d = self.d_i_m / self.d_o_m
        self.eps = self.abseps / self.d_o_m

        
    def calc_PVT(self, p, t):
        """
        Метод для расчета PVT-модели 
        :param p: текущее давление, Па 
        :param t: текущая температура, К
        """
        self.rp = self.qu_gas_m3sec / self.qu_liq_m3sec
        self.pvt_model =  {"black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
                                         "rp": self.rp,
                                         "oil_correlations":
                                          {"pb": "Standing", "rs": "Standing",
                                           "rho": "Standing","b": "Standing",
                                          "mu": "Beggs", "compr": "Vasquez"},
                            "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
                                                  "z": "Dranchuk", "mu": "Lee"},
                             "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                    "rho": "Standing", "mu": "McCain"},
                            "rsb": {"value": 50, "p": 10000000, "t": 303.15},
                             "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
                             "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
                             "table_model_data": None, "use_table_model": False}}
        self.PVT = FluidFlow(self.qu_liq_m3sec, self.wct, self.pvt_model)
        self.PVT.calc_flow(p, t)
        self.mu_gas_pasec = self.PVT.mug
        self.mu_liq_pasec = self.PVT.mul
        self.rho_gas_kgm31 = self.PVT.rho_gas
        # self.rho_liq_kgm3 = self.PVT.rho_wat
        self.rho_liq_kgm3 = self.PVT.rho_oil
        self.sigma_Nm = self.PVT.stlg

        
    def _mixture_velocity_Caetano(self, initial_f):
        """
        Метод для определения критической скорости dispersed bubbly flow
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        """
        right_part = 2 * (initial_f ** 1.2) * (self.friction_coeff** 0.4) * ((2 / self.d_equ_m) **
                     0.4) * ((self.rho_liq_kgm3 / self.sigma_Nm) ** 0.6 ) *(0.4 * self.sigma_Nm / (
                         (self.rho_liq_kgm3 - self.rho_gas_kgm31) * CONST.g) ** 0.5)

        left_part = 0.725 + 4.15 * (self.vs_gas_msec / initial_f) ** 0.5

        return right_part - left_part

    def friction_coefv2(self, rho):
        """
        Метод для определения коэффициента трения
        :param rho: фактическая плотность ГЖС

        :return: коэффициент трения
        """
        self.number_Re_s = rho * self.v_mix_msec * self.d_equ_m / self.mu_mix_pasec
        self.frict = fr.Friction(self.d_o_m)
        self.ff =self.frict.calc_norm_ff(self.number_Re_s,self.eps,1)

        return self.ff

    def calc_pattern(self):
        """
        Метод для расчета критических значений скоростей
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        Upward Vertical Two-Phase Flow Through an Annulus—Part I for to dispersed
        """
        #bubble to slug transition [4]
        self.vs_gas_bubble2slug_msec = ((self.C0 * self.vs_liq_msec + self.v_d_msec) / (4 
                                    - self.C0)) * np.sin(self.theta * np.pi/180)

        #to annular transirion [17]
        self.vs_gas_2annular_msec = 3.1 * (self.sigma_Nm * CONST.g * (self.rho_liq_kgm3 - 
                                    self.rho_gas_kgm31) / (self.rho_gas_kgm31) ** 2) ** 0.25

        #bubble/slug to dispersed transition [6]

        self.friction_coeff = self.friction_coefv2(self.rho_m_rash_kgm3)
        self.v_m_krit2disp_msec = float(sp.fsolve(self._mixture_velocity_Caetano, 1000))

        self.set_flow_pattrn()

    def set_flow_pattrn(self):
        """
        Метод для определения структуры потока
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
        Метод для расчета истинной объемной концентрации газа в bubbly flow
        """
        self.v_d_msec = (1.53 * (CONST.g * self.sigma_Nm * (self.rho_liq_kgm3 - self.rho_gas_kgm31)
                     / (self.rho_liq_kgm3)**2 ) ** 0.25) #3

        self.v_gas_msec = self.C0 * self.v_mix_msec + self.v_d_msec #1
        self.epsi = self.vs_gas_msec / self.v_gas_msec #2

    def calc_slug_churn(self, C) -> float:
        """
        Метод для расчета истинной объемной концентрации газа в slug и churn flow
        :param С: параметр распределения газовой фазы в потоке
        """
        self.v_d_msec = (1.53 * (CONST.g * self.sigma_Nm * (self.rho_liq_kgm3 - self.rho_gas_kgm31) 
                        / (self.rho_liq_kgm3)**2 ) ** 0.25) #3

        self.v_dt_msec = (0.345 + 0.1 * (self.d_i_m / self.d_o_m)) *((CONST.g * self.d_o_m * (
                        self.rho_liq_kgm3 - self.rho_gas_kgm31)/(self.rho_gas_kgm31)) ** 0.5) #14

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
        Метод для вычисления фактический длины пленки жидкости в slug/churn
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
        Метод для нахождения градиента давления на ускорения в churn и slug flow
        [20,23,38,44,47] 

        :return: градиент давления на ускорение
        """
        self.h_ls = 1 - self.epsi_s
        self.h_lf = 1 - self.epsi_t
        self.len_ls = 16 * self.d_equ_m #38
        self.len_su = self.len_ls / self.len_s_m
        self.v_lls = ((self.vs_liq_msec + self.vs_gas_msec) - 1.53 * ((self.rho_liq_kgm3 - self.rho_gas_kgm31) #23
                    * CONST.g * self.sigma_Nm / (self.rho_liq_kgm3) ** 2) ** 0.25 * self.h_ls ** 0.5 * (1 - self.h_ls)) 
        self.v_gls = (1.53 * ((self.rho_liq_kgm3 - self.rho_gas_kgm31) * CONST.g * self.theta / (self.rho_liq_kgm3) #20
                        ** 2) ** 0.25 * self.h_ls ** 0.5) - self.v_lls
        self.act_len_lf = float(sp.fsolve(self._actual_film_length, 0.005))
        self.v_llf = (CONST.g * 2 * self.act_len_lf) ** 0.5 - self.v_dt_msec #47
        self.grad_p_acc = (self.rho_liq_kgm3 * (self.h_lf / self.len_su) * (self.v_llf - self.v_dt_msec) 
                    * (self.v_llf - self.v_lls))
        return self.grad_p_acc

    def _acceler_grad_p_annular(self):
        """
        Метод для расчета потерь на ускорения в кольцевом режиме потока
        """
        self.v_dt_msec = (0.345 + 0.1 * (self.d_i_m / self.d_o_m)) *((CONST.g * self.d_o_m * (
                        self.rho_liq_kgm3 - self.rho_gas_kgm31)/(self.rho_gas_kgm31)) ** 0.5)
        self.len_su= 1
        self.act_len_lf = self.len_su
        self.v_llf = (CONST.g * 2 * self.act_len_lf) ** 0.5 - self.v_dt_msec
        grad_p_acc_an = (self.rho_liq_kgm3 * (self.hl_total / self.len_su) * (self.v_llf - self.v_dt_msec) 
                    * self.v_llf)
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

    def calc_hl_total_annular(self):
        """
        Метод для вычисления концентрации жидкости в потоке кольцевой структуры[77]
        Допустил, что толщина пленки жидкости на внешней(или внутренней) трубе известна()
        """
        self.delta_o = 0.005
        self.delta_i = self.delta_o * self._ratio_t()
        self.phi = (10 ** 4 * self.vs_gas_msec * self.mu_gas_pasec / self.sigma_Nm * (self.rho_gas_kgm31 #[79]
                    / self.rho_liq_kgm3) ** 0.5)
        self.fe = 1 - m.exp((self.phi - 1.5) * (-0.125))
        self.hl_total = (4 / ((self.d_o_m) * (1 - self.k_ratio_d ** 2)) * (self.delta_o * (1 - self.delta_o / self.d_o_m) #[77]
                        + self.delta_i * self.k_ratio_d * (1 + self.delta_i / self.d_i_m) + self.vs_liq_msec * self.fe 
                        / ((self.vs_liq_msec * self.fe + self.vs_gas_msec) * (1 - self.k_ratio_d ** 2)) * 
                        (1 - self.k_ratio_d ** 2 - 4 * self.delta_o / self.d_o_m * (1 - self.delta_o / self.d_o_m) 
                        - 4 *self.delta_i *self.k_ratio_d / self.d_o_m * (1 + self.delta_i / self.d_i_m))))
        self.epsi = 1 - self.hl_total 
          

    def calc_rho_mix(self):
        """
        Метод для расчета плотности смеси
        """
        if self.flow_pattern == 0 or self.flow_pattern == 1:
            self.calc_bubbly()
        elif self.flow_pattern == 2:
            self.calc_slug_churn(self.C0)
        elif self.flow_pattern == 3: #tut3
            self.calc_slug_churn(self.C1)
        elif self.flow_pattern == 4:
            self.calc_hl_total_annular()
        
        self.rho_mix_kgm3 = self.rho_liq_kgm3 * (1 - self.epsi) + self.rho_gas_kgm31 * self.epsi

    def calc_pressure_gradient(self, p, t):
        """
        Метод для расчета градиента давления
        Upward Vertical Two-Phase Flow Through an Annulus—Part II
        :param p: текущее давление, Па 
        :param t: текущая температура, К

        :return: суммарный градиент давления
        
        """
        self.calc_PVT(p, t)
        self.calc_rash()
        self.calc_pattern()
        self.calc_rho_mix()

        if self.flow_pattern == 0: #[5-14]
            self.density_grad_pam = self.rho_mix_kgm3 * CONST.g * np.sin(self.theta * np.pi/180)

            self.friction_coeff_s = self.friction_coefv2(self.rho_mix_kgm3)
            self.friction_grad_pam = (4 * self.friction_coeff_s / (self.d_o_m - self.d_i_m) 
                                     * self.v_mix_msec ** 2 / 2)

            self.acceleration_grad_pam = 0

        elif self.flow_pattern == 1: #[15-16]
            self.density_grad_pam = self.rho_mix_kgm3 * CONST.g * np.sin(self.theta * np.pi/180)

            self.friction_coeff_s = self.friction_coefv2(self.rho_mix_kgm3)
            self.friction_grad_pam = (4 * self.friction_coeff_s / (self.d_o_m - self.d_i_m) 
                                     * self.v_mix_msec ** 2 / 2)

            self.acceleration_grad_pam = 0
            
        elif self.flow_pattern == 2 or self.flow_pattern == 3: #предположил что для slug и churn одна методика. Концентрацию воды нашел как 1 - epsi
            self.rho_slug_kgm3 = self.rho_gas_kgm31 * self.epsi_s + self.rho_liq_kgm3 * (1-self.epsi_t) # В соответствии c [51] 

            self.density_grad_pam = self.rho_slug_kgm3 * CONST.g * self.len_s_m #[50]

            self.friction_coeff_s = self.friction_coefv2(self.rho_slug_kgm3)
            self.friction_grad_pam = ((2 * self.friction_coeff_s / self.d_equ_m * self.rho_slug_kgm3) #[53]
                                     * (self.vs_gas_msec + self.vs_liq_msec) **2 * self.len_s_m)

            self.acceleration_grad_pam = self._acceler_grad_p() 

        elif self.flow_pattern == 4:# над ускорением подумать
            self.density_grad_pam = self.rho_mix_kgm3 * CONST.g * np.sin(self.theta * np.pi/180)

            self.friction_coeff_s = self.friction_coefv2(self.rho_mix_kgm3)
            self.friction_grad_pam = (4 * self.friction_coeff_s / (self.d_o_m - self.d_i_m) 
                                     * self.v_mix_msec ** 2 / 2)

            self.acceleration_grad_pam = self._acceler_grad_p_annular()

        self.result_grad_pam = self.friction_grad_pam  + self.density_grad_pam + self.acceleration_grad_pam

        return self.result_grad_pam

    def grad_func(self, h, pt):
        """
        Функция для интегрирования 
        :param pt: давление и температура, Па,К
        :h: переменная интегрирования
        """ 
        dp_dl = self.calc_pressure_gradient(pt[0], pt[1]) 
        dt_dl = 0.03
        return dp_dl, dt_dl

    def func_p_list(self):
        """
        Метод для интегрирования градиента давления(расчет сверху вниз)
        :return: распределение давления и температуры по стволу скважины
        """
        p0,t0 = self.p_head, self.t_head
        h0 = 0
        h1 = self.h
        self.calc_PVT(p0, t0)
        self.calc_rash()
        self.calc_pattern()
        self.calc_rho_mix()
        steps = [i for i in range(h0, h1+50, 50)]
        sol = solve_ivp(self.grad_func, t_span=(h0, h1), y0=[p0, t0], t_eval = steps) 
        return sol.y, 




if __name__ == '__main__':

    #ТЕСТ
    # test2 = HasanKabirAnn(qu_gas_m3day=0,qu_liq_m3day=400)
    # print(test2.flow_pattern_name)
    # print(test2.func_p_list()) #хорошая сходимость
    
    # qg = [i for i in range(0, 1000, 100)]
    # ql = [i for i in range(0, 1500, 100)]
    # for i in qg:
    #     flow = HasanKabirAnn(qu_gas_m3day=i,qu_liq_m3day=400)
    #     print(flow.flow_pattern_name)
    #     print(flow.func_p_list())

        # print(flow.func_p_list())
        # flow.func_p_list()
        

    #TEST тестовый тех режим 
  
    #2 h=2516 p0=60 pk=96 ql=62 qg=124060 d_i=73 d_o=157-15 фонтан
 
    test = HasanKabirAnn(qu_gas_m3day = 124060, qu_liq_m3day = 62 , p_head = 60, d_i_m = 73, d_o_m = 142, h = 2400, wct=0.14)

    print(test.func_p_list())
    print(test.flow_pattern_name)    
      
      