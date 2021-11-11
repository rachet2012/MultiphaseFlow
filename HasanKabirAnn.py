import warnings
import math as m
import numpy as np
import pandas as pd
import scipy.constants as CONST
import scipy.optimize as sp
from scipy.integrate import solve_ivp
from unifloc.pvt.fluid_flow import FluidFlow
import unifloc.pipe._friction as fr
import unifloc.common.trajectory as tr
import unifloc.pipe._pipe as pipe
import scipy.interpolate as interp
import unifloc.common.ambient_temperature_distribution as amb

warnings.filterwarnings("ignore",category=RuntimeWarning)


#TODO fsolve(экспоненты и логарифм, пока подходит только fsolve)
class HasanKabirAnn():
    """
    Класс для расчета градиента давления в затрубном пространстве по корреляции HasanKabir/CaetanoBrill
    Определяются структура потока, истинная концентрация газа, плотность смеси,
    градиенты на гравитацию, трение, ускорение.

    """

    def __init__(self, fluid ,  d_i_m: float = 73, d_o_m: float = 142,
         theta_deg: float = 90, abseps:float = 2.54) -> None:
        """
        :param fluid: PVT модель флюида
        :param d_i_m: внешний диаметр НКТ, мм
        :param d_o_m: внутренний диаметр ЭК, мм
        :param theta: угол наклона скважины, градусы
        :param abseps: абсолютная шероховатость стенок трубы, м^10-5

        """
        self.fluid = fluid

        self.d_i_m = d_i_m / 1000
        self.d_o_m = d_o_m / 1000
        self.angle = theta_deg
        self.abseps = abseps / 100000

        self.flow_pattern = None
        self.flow_pattern_name = None

        self.hl = None
        self.epsi = None
        self.rho_mix = None
        self.dp_dl = None
        self.dp_dl_acc = None
        self.dp_dl_gr = None
        self.dp_dl_fr = None

    def _calc_par(self):
        """
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Hasan Kabir

        :param f_m2: площадь сечения затрубного пространства, м2
        :param vs_gas_msec: приведенная скорость газа, м/с
        :param vs_liq_msec: приведенная скорость жидкости, м/с
        :param v_mix_msec: приведенная скорость смеси, м/с
        :param _d_equ_m: эквивалентный диаметр затрубного пространства, м2
        """
        f_m2 =  CONST.pi * ((self.d_o_m/2)**2 - (self.d_i_m/2)**2)
        self.vsg = self.fluid.qg / f_m2
        self.vsl = self.fluid.ql / f_m2
        self.vsm = (self.fluid.qg + self.fluid.ql) / f_m2
        self._d_equ_m = self.d_o_m - self.d_i_m


    def _mixture_velocity_Caetano(self, initial_f):
        """
        Метод для определения критической скорости dispersed bubbly flow
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        """
        rho_m_rash_kgm3 = (self.fluid.qg / (self.fluid.qg + self.fluid.ql)
                                * self.fluid.rg + (1 - self.fluid.qg / (self.fluid.qg +
                                self.fluid.ql)) * self.fluid.rl)
        friction_coeff = self._actual_friction_coef(rho_m_rash_kgm3)
        right_part = 2 * (initial_f ** 1.2) * (friction_coeff** 0.4) * ((2 / self._d_equ_m) **
                     0.4) * ((self.fluid.rl/ self.fluid.stlg) ** 0.6 ) *(0.4 * self.fluid.stlg / (
                         (self.fluid.rl- self.fluid.rg) * CONST.g) ** 0.5)
        left_part = 0.725 + 4.15 * (self.vsg / initial_f) ** 0.5
        return right_part - left_part

    def _friction_coefficient_Gunn_Darling(self, initial_ff):
        """
        Метод для определения коэффициента трения в турбулентном течении
        Upward Vertical Two-Phase Flow Through an Annulus—Part I [15-27]
        """
        right_part = (4 * m.log(self.n_re* (initial_ff * (16 / self.fca) **
                     (0.45 * m.exp(-(self.n_re - 3000) / (10 ** 6)))) ** 0.5) - 0.4)
        left_part = 1 / (initial_ff * (16 / self.fca) ** (0.45 * m.exp(-(self.n_re -
                    3000) / 10 ** 6))) ** 0.5
        return right_part - left_part

    def _calc_pattern(self):
        """
        Метод для расчета критических значений скоростей
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        Upward Vertical Two-Phase Flow Through an Annulus—Part I for to dispersed
        """
        #bubble to slug transition [4]
        v_d_msec = 1.53 * (CONST.g * self.fluid.stlg * (self.fluid.rl
                        - self.fluid.rg) / (self.fluid.rl)**2 ) ** 0.25
        self.vs_gas_bubble2slug_msec = ((1.2 * self.vsl + v_d_msec) / (4
                                    - 1.2)) * np.sin(self.angle * np.pi/180)
        #to annular transirion [17]
        self.vs_gas_2annular_msec = ((3.1 * (self.fluid.stlg * CONST.g * (self.fluid.rl-
                                    self.fluid.rg) / (self.fluid.rg) ** 2) ** 0.25 + 1)
                                    * np.sin(self.angle * np.pi/180))
        #bubble/slug to dispersed transition [6]
        self.v_m_krit2disp_msec = sp.fsolve(self._mixture_velocity_Caetano, 6, maxfev=13)
        self._set_flow_pattrn()

    def _actual_friction_coef(self, rho):
        """
        Метод для расчета коэффициента трения
        :param rho: плотность ГЖС, кг/м3

        :return: коэффициент трения, безразмерн.
        """
        k_ratio_d = self.d_i_m / self.d_o_m
        frict = fr.Friction(self._d_equ_m)
        mu_mix_pasec = (self.vsl / self.vsm * self.fluid.mul
                            + self.vsg / self.vsm * self.fluid.mug)
        self.n_re = frict.calc_n_re(rho, self.vsm, mu_mix_pasec)
        self.fca = (16 * (1 - k_ratio_d) ** 2 /
                    ((1 - k_ratio_d ** 4) / (1 - k_ratio_d ** 2) -
                     (1 - k_ratio_d ** 2) / m.log(1 / k_ratio_d)))
        if self.n_re < 3000:  # laminar flow
            fric = self.fca / self.n_re
        else:  # turbulent flow
            fric = sp.fsolve(self._friction_coefficient_Gunn_Darling, 0.021, maxfev=13)
        return fric

    def _set_flow_pattrn(self):
        """
        Метод для определения структуры потока
        """
        if self.vsg <= self.vs_gas_bubble2slug_msec:
            self.flow_pattern = 0
            self.flow_pattern_name = 'Bubble flow pattern - пузырьковый режим'
        elif (self.vsg >= self.vs_gas_bubble2slug_msec and (0.25 * self.vsg) < 0.52
                and self.vsm < self.v_m_krit2disp_msec):
            self.flow_pattern = 2
            self.flow_pattern_name = 'Slug flow pattern - Пробковый режим'
        elif self.vsg >= self.vs_gas_bubble2slug_msec and (0.25 * self.vsg) >= 0.52 :
            self.flow_pattern = 3
            self.flow_pattern_name = 'Chug flow pattern - Вспененный режим'
        elif self.vsm >= self.v_m_krit2disp_msec:
            self.flow_pattern = 1
            self.flow_pattern_name = 'Dispersed bubble flow pattern - дисперсионно-пузырьковый режим'
        elif self.vsg >= self.vs_gas_2annular_msec :
            self.flow_pattern = 4
            self.flow_pattern_name = 'Annular flow pattern - Кольцевой режим'

    def _calc_bubbly(self) -> float:
        """
        Метод для расчета истинной объемной концентрации газа в bubbly flow
        """
        v_d_msec = (1.53 * (CONST.g * self.fluid.stlg * (self.fluid.rl- self.fluid.rg)
                     / (self.fluid.rl)**2 ) ** 0.25) #3
        v_gas_msec = 1.2 * self.vsm + v_d_msec #1
        self.epsi = self.vsg / v_gas_msec #2
        self.hl = 1 - self.epsi

    def _calc_slug_churn(self) -> float:
        """
        Метод для расчета истинной объемной концентрации газа в slug и churn flow
        """
        self.v_dt_msec = (1.2 *(self.vsg + self.vsl) + 0.345 *
                        (CONST.g * (self.d_i_m + self.d_o_m)) ** 0.5
                        * np.sin(self.angle * np.pi/180) ** 0.5
                        * (1 + np.cos(self.angle * np.pi/180)) ** 1.2)#17
        self.epsi_s = self.vsg / (1.2 * self.vsm + self.v_dt_msec)
        self.epsi_t = self.vsg / (1.15 * self.vsm + self.v_dt_msec) #7
        if self.vsg > 0.4:
            self.len_s_m = 0.1 / self.epsi_s
            self.epsi = (1 - self.len_s_m) * self.epsi_t + 0.1  #9a
        else:
            self.len_s_m = 0.25 * self.vsg / self.epsi_s
            self.epsi = (1 - self.len_s_m) * self.epsi_t + 0.25 * self.vsg #9b
        self.hl = 1 - self.epsi

    def _actual_film_length(self):
        """
        Метод для вычисления фактический длины пленки жидкости в slug/churn
        Уравнение  [44]

        :return: фактическую длину пленки жидкости в slug/churn, м
        """
        coef_b = (-2 * (1 - self.vsg / self.v_dt_msec) * ((self.vsg - self.v_gls #45
                    *(1 - self.h_ls)) / self.v_dt_msec) * self.len_ls + (2 / CONST.g) * (self.v_dt_msec
                    - self.v_lls) ** 2 * self.h_ls **2) / (1 - self.vsg / self.v_dt_msec) ** 2
        coef_c = (((self.vsg - self.v_gls * (1 - self.h_ls)) / self.v_dt_msec * self.len_ls)
                        / (1 - self.vsg / self.v_dt_msec)) ** 2
        discr = (coef_b ** 2 - 4 * coef_c)
        #дискриминант отрицательный, уравнение в каких то диапазонах не решается
        if discr > 0 :
            x1 = (-coef_b + m.sqrt(discr)) / 2
            x2 = (-coef_b - m.sqrt(discr)) / 2
            if x1 >= 0 and x2 < 0:
                resh = x1
            elif x2 >= 0 and x1 <0:
                resh = x2
            elif x1 < 0 and x2 < 0:
                resh = 0.000001
            elif x1 > x2:
                resh = x2
            elif x1 < x2:
                resh = x1
        elif discr == 0:
            resh = -coef_b / 2
        else:
            resh = 0.000001
        return resh

    def _acceler_grad_p(self) :
        """
        Метод для нахождения градиента давления на ускорения в churn и slug flow
        [20,23,38,44,47]

        :return: градиент давления на ускорение в churn и slug flow, Па/м
        """
        self.h_ls = 1 - self.epsi_s
        h_lf = 1 - self.epsi_t
        self.len_ls = 16 * self._d_equ_m #38
        len_su = self.len_ls / self.len_s_m
        self.v_lls = ((self.vsl + self.vsg) - 1.53 * ((self.fluid.rl- self.fluid.rg) #23
                    * CONST.g * self.fluid.stlg / (self.fluid.rl**2)) ** 0.25 * self.h_ls ** 0.5 * (1 - self.h_ls))
        self.v_gls = ((1.53 * ((self.fluid.rl- self.fluid.rg) * CONST.g * self.fluid.stlg / (self.fluid.rl**2) #20
                        ) ** 0.25 * self.h_ls ** 0.5) + self.v_lls)
        act_len_lf = (self._actual_film_length())
        v_llf = m.fabs((CONST.g * 2 * act_len_lf) ** 0.5 - self.v_dt_msec) #47
        grad_p_acc = (self.fluid.rl* (h_lf / len_su) * (v_llf - self.v_dt_msec)
                    * (v_llf - self.v_lls))
        if grad_p_acc < 0:
            grad_p_acc_res = 0
        else:
            grad_p_acc_res = grad_p_acc
        return grad_p_acc_res

    def _acceler_grad_p_annular(self):
        """
        Метод для расчета потерь на ускорения в кольцевом режиме потока

        :return: градиент давления на ускорение в кольцевом режиме потока, Па/м
        """
        v_dt_msec = (0.345 + 0.1 * (self.d_i_m / self.d_o_m)) *((CONST.g * self.d_o_m * (
                        self.fluid.rl- self.fluid.rg)/(self.fluid.rg)) ** 0.5)
        len_su = 1
        act_len_lf = len_su
        v_llf = (CONST.g * 2 * act_len_lf) ** 0.5 - v_dt_msec
        grad_p_acc_an = (self.fluid.rl* (self.hl / len_su) * (v_llf - v_dt_msec)
                    * v_llf)
        return grad_p_acc_an

    def _calc_hl_annular(self):
        """
        Метод для вычисления концентрации газа в потоке кольцевой структуры[77]
        Допустил, что толщина пленки жидкости на внешней(или внутренней) трубе известна
        """
        delta_o = 0.005
        k_ratio_d = self.d_i_m / self.d_o_m
        angle_wt_average = (1 / (1 - k_ratio_d ** 2) * (2 * m.asin(k_ratio_d) + 2 * #[88]
                             k_ratio_d  * (1 - k_ratio_d ** 2) ** 0.5 - CONST.pi *
                             k_ratio_d ** 2))
        t_ratio = angle_wt_average / ((2 * CONST.pi - angle_wt_average) * k_ratio_d)
        delta_i = delta_o * t_ratio
        phi = (10 ** 4 * self.vsg * self.fluid.mug / self.fluid.stlg * (self.fluid.rg#[79]
                    / self.fluid.rl) ** 0.5)
        fe = 1 - m.exp((phi - 1.5) * (-0.125))
        self.hl = (4 / ((self.d_o_m) * (1 - k_ratio_d ** 2)) * (delta_o * (1 - delta_o / self.d_o_m)#[77]
                        + delta_i * k_ratio_d * (1 + delta_i / self.d_i_m) + self.vsl * fe
                        / ((self.vsl * fe + self.vsg) * (1 - k_ratio_d ** 2)) *
                        (1 - k_ratio_d ** 2 - 4 * delta_o / self.d_o_m * (1 - delta_o / self.d_o_m)
                        - 4 * delta_i * k_ratio_d / self.d_o_m * (1 + delta_i / self.d_i_m))))
        self.epsi = 1 - self.hl

    def calc_rho_mix(self):
        """
        Метод для расчета плотности смеси
        """
        self._calc_pattern()
        if self.flow_pattern == 0 or self.flow_pattern == 1:
            self._calc_bubbly()
        elif self.flow_pattern == 2 or self.flow_pattern == 3:
            self._calc_slug_churn()
        elif self.flow_pattern == 4:
            self._calc_hl_annular()
        self.rho_mix_kgm3 = self.fluid.rl * (1 - self.epsi) + self.fluid.rg * self.epsi

    def calc_grav(self):
        if self.flow_pattern == 0 or self.flow_pattern == 1 or self.flow_pattern == 4: #[5-14]
            self.dp_dl_gr = self.rho_mix_kgm3 * CONST.g * np.sin(self.angle * np.pi/180)
        elif self.flow_pattern == 2 or self.flow_pattern == 3 :
            self.rho_slug_kgm3 = self.fluid.rg * self.epsi_s + self.fluid.rl* (1 - self.epsi_s) # В соответствии c [51]
            self.dp_dl_gr = self.rho_slug_kgm3 * CONST.g  * self.len_s_m *  np.sin(self.angle * np.pi/180)
        return self.dp_dl_gr

    def calc_fric(self):
        if self.flow_pattern == 0 or self.flow_pattern == 1 or self.flow_pattern == 4: #[5-14]
            friction_coeff_s = self._actual_friction_coef(self.rho_mix_kgm3)
            self.dp_dl_fr = ((4 * friction_coeff_s / (self._d_equ_m )
                                     * self.vsm ** 2 / 2) * self.rho_mix_kgm3)
        elif self.flow_pattern == 2 or self.flow_pattern == 3 :
            self.rho_slug_kgm3 = self.fluid.rg * self.epsi_s + self.fluid.rl* (1 - self.epsi_s) # В соответствии c [51]
            friction_coeff_s = self._actual_friction_coef(self.rho_slug_kgm3)
            self.dp_dl_fr = ((2 * friction_coeff_s / self._d_equ_m * self.rho_slug_kgm3)#[53]
                                     * (self.vsg + self.vsl) **2 * self.len_s_m)
        return self.dp_dl_fr

    def calc_acc(self):
        if self.flow_pattern == 0 or self.flow_pattern == 1: #[5-14]
            self.dp_dl_acc = 0
        elif self.flow_pattern == 2 or self.flow_pattern == 3 :
            self.rho_slug_kgm3 = self.fluid.rg * self.epsi_s + self.fluid.rl* (1 - self.epsi_s) # В соответствии c [51]
            self.dp_dl_acc = self._acceler_grad_p()
        elif self.flow_pattern == 4: #методику не приводят
            self.dp_dl_acc = self._acceler_grad_p_annular()
        return self.dp_dl_acc

    def calc_grad(self):
        """
        Метод для расчета градиента давления
        Upward Vertical Two-Phase Flow Through an Annulus—Part II

        :return: суммарный градиент давления, Па/м

        """
        self._calc_par()
        self.calc_rho_mix()

        self.dp_dl_gr = self.calc_grav()
        self.dp_dl_fr = self.calc_fric()
        self.dp_dl_acc = self.calc_acc()

        self.dp_dl = self.dp_dl_fr  + self.dp_dl_gr + self.dp_dl_acc
        return self.dp_dl


if __name__ == '__main__':

    def grad_func(h, pt, PVT, traj, corr,d_o,d_i,md_tvd):
        """
        Интегрируемае функция

        :param h: текущая глубина, м
        :param pt: текущее давление, Па и текущая температура, К
        :param PVT: объект с PVT моделью
        :param traj: объект с инклинометрией
        :param corr: объект с корреляцией
        :param d_o: внутренний диаметр ЭК, мм. Таблица формата pd.DataFrame
        :param d_i: внешний диаметр НКТ, мм. Таблица формата pd.DataFrame
        :param md_tvd: инклинометрия скважины, м. Таблица формата pd.DataFrame

        :return: градиент давления в заданной точке трубы
        при заданных термобарических условиях, Па/м
        :return: градиент температуры в заданной точке трубы
        при заданных термобарических условиях, К/м
        """
        md_steps = [0]

        md = interp.interp1d(
                md_tvd["TVD"], md_tvd["MD"], fill_value="extrapolate", kind="previous"
                )(h)
        md_prev = md_steps[-1]
        md_steps.append(md)
        h_steps = [0]
        h_prev = h_steps[-1]
        h_steps.append(h)
        theta = traj.calc_angle(h_prev,h)
        # print(md)
        PVT.calc_flow(pt[0],pt[1])
        corr.theta = theta
        corr.fluid = PVT
        corr.d_o_m = interp.interp1d(
                d_o["MD"], d_o["d_o"], fill_value="extrapolate", kind="previous"
                )(md)
        corr.d_i_m =interp.interp1d(
                d_i["MD"], d_i["d_i"], fill_value="extrapolate", kind="previous"
                )(md)
        # print(corr.d_o_m)
        dp_dl = corr.calc_grad()
        dt_dl = 0.03
        return dp_dl, dt_dl

    def func_p_list(p_head, t_head, h, PVT, traj, corr,d_o, d_i,md_tvd):
        """
        Функция для интегрирования давления, температуры в трубе

        :param p_head: давление на устье, Па
        :param t_head: температура на устье, К
        :param h: граничная глубина, м
        :param PVT: объект с PVT моделью
        :param traj: объект с инклинометрией
        :param corr: объект с корреляцией
        :param d_o: внутренний диаметр ЭК, мм. Таблица формата pd.DataFrame
        :param d_i: внешний диаметр НКТ, мм. Таблица формата pd.DataFrame
        :param md_tvd: инклинометрия скважины, м. Таблица формата pd.DataFrame


        :return: массив температур, К, массив давлений, Па
        """
        p0,t0 = p_head, t_head
        h0 = 0
        h1 = h
        steps = [i for i in range(h0, h1+50, 50)]
        sol = solve_ivp(grad_func,
            t_span=(h0, h1),
            y0=[p0, t0],
            t_eval=steps,
            max_step = 55,
            args=(
            PVT,
            traj,
            corr,
            d_o,
            d_i,
            md_tvd)
        )
        return sol.y,

    def schet(rp,qu_liq_r,wct_r,p_head_r,t_head_r , absep_r, gamma_gas, gamma_wat, gamma_oil, pb,
         t_res, rsb, muob, bob, md1, md2, md3, tvd1, tvd2, tvd3, d_o_1, d_o_2, d_o_3, d_i_1, d_i_2, d_i_3):
        """
        Функция для инициилизации расчета
        :param rp: ГФ, м3/м3
        :param qu_liq_r: дебит жидкости, м3/сут
        :param wct_r: обводненнность продукции, дол.ед
        :param p_head_r: давление на устье, Па
        :param t_head_r: температура на устье, К
        :param absep_r: абсолютная шероховатость стенок трубы, м*10^-5
        :param gamma_gas: относительная плотность газа, дол.ед
        :param gamma_wat: относительная плотность воды, дол.ед
        :param gamma_oil: относительная плотность нефти, дол.ед
        :param pb: давление насыщения, Па
        :param t_res: пластовая температура, К
        :param rsb: калибровочное значение газо-ния при дав-ии нас-я, ст. м3 газа/ст. м3 нефти
        :param muob: калибровочное значение вязкости нефти, сПз
        :param bob: калибровочное значение объемного коэффициента нефти, ст.м3/ст.м3
        :param md1,md2,md3: точки md инклинометрии, м
        :param tvd1,tvd2,tvd3: точки tvd инклинометрии, м
        :param d_i_1, d_i_2, d_i_3: внешний диаметр НКТ в соотв.точках md, мм
        :param d_o_1, d_o_1, d_o_1: внутренний диаметр ЭК в соотв.точках md, мм
        :param PVT: объект с PVT моделью
        :param traj: объект с инклинометрией

        :return: забойное давление, атм
        """
        pvt_model_data = {"black_oil": {"gamma_gas": gamma_gas, "gamma_wat": gamma_wat, "gamma_oil": gamma_oil,
                                         "rp": rp,
                                         "oil_correlations":
                                          {"pb": "Standing", "rs": "Standing",
                                           "rho": "Standing","b": "Standing",
                                           "mu": "Beggs", "compr": "Vasquez"},
                             "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
                                                  "z": "Dranchuk", "mu": "Lee"},
                             "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                    "rho": "Standing", "mu": "McCain"},
                             "rsb": {"value": rsb, "p": pb, "t": t_res},
                             "muob": {"value":muob, "p": pb, "t": t_res},
                             "bob": {"value": bob, "p": pb, "t": t_res},
                             "table_model_data": None, "use_table_model": False}}
        pvt = FluidFlow(qu_liq_r/86400, wct_r, pvt_model_data)
        md_tvd = pd.DataFrame(columns=["MD", "TVD"],
                                        data=[[0, 0], [md1, tvd1],
                                        [md2, tvd2], [md3, tvd3]])
        trajectory = tr.Trajectory(md_tvd)
        d_o = pd.DataFrame(columns=["MD", "d_o"],
                                    data=[[0, d_o_1], [md1, d_o_1],
                                    [md2, d_o_2], [md3, d_o_3]])
        d_i = pd.DataFrame(columns=["MD", "d_i"],
                                    data=[[0, d_i_1], [md1, d_i_1],
                                    [md2, d_i_2], [md3, d_i_3]])
        test3 = HasanKabirAnn(d_i_m = d_i_1, d_o_m = d_o_1, fluid = pvt, abseps= absep_r)
        vr = func_p_list(p_head = p_head_r, t_head=t_head_r, h=tvd3,
                         PVT=pvt, traj=trajectory, corr = test3, d_o=d_o, d_i=d_i,md_tvd=md_tvd)
        vr1 = vr[0]
        vr2 = vr1[0]
        vr3= vr2[-1]
        return vr3/101325

    def schet_pipe(rp,qu_liq_r,wct_r,p_head_r,t_head_r , absep_r, gamma_gas, gamma_wat, gamma_oil, pb,
         t_res, rsb, muob, bob, md1, md2, md3, tvd1, tvd2, tvd3, d_o_1, d_o_2, d_o_3, d_i_1, d_i_2, d_i_3):
        pvt_model_data = {"black_oil": {"gamma_gas": gamma_gas, "gamma_wat": gamma_wat, "gamma_oil": gamma_oil,
                                         "rp": rp,
                                         "oil_correlations":
                                          {"pb": "Standing", "rs": "Standing",
                                           "rho": "Standing","b": "Standing",
                                           "mu": "Beggs", "compr": "Vasquez"},
                             "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
                                                  "z": "Dranchuk", "mu": "Lee"},
                             "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                    "rho": "Standing", "mu": "McCain"},
                             "rsb": {"value": rsb, "p": pb, "t": t_res},
                             "muob": {"value":muob, "p": pb, "t": t_res},
                             "bob": {"value": bob, "p": pb, "t": t_res},
                             "table_model_data": None, "use_table_model": False}}
        pvt = FluidFlow(qu_liq_r/86400, wct_r, pvt_model_data)
        md_tvd = pd.DataFrame(columns=["MD", "TVD"],
                                        data=[[0, 0], [md1, tvd1],
                                        [md2, tvd2], [md3, tvd3]])
        trajector = tr.Trajectory(md_tvd)
        d_o = pd.DataFrame(columns=["MD", "d_o"],
                                    data=[[0, d_o_1], [md1, d_o_1],
                                    [md2, d_o_2], [md3, d_o_3]])
        d_i = pd.DataFrame(columns=["MD", "d_i"],
                                    data=[[0, d_i_1], [md1, d_i_1],
                                    [md2, d_i_2], [md3, d_i_3]])
        ambient_temperature_data = {"MD": [0, md3], "T": [303.15, 363.15]}
        amb_temp = amb.AmbientTemperatureDistribution(ambient_temperature_data)
        step = [i for i in range(0, tvd3+50, 50)]
        pip = pipe.Pipe(fluid=pvt, d = d_i, roughness=2.54,hydr_corr_type='HasanKabir')
        return pip.integrate_pipe(p0 = p_head_r,t0= t_head_r,h0=0,h1=tvd3,trajectory= trajector,
                 amb_temp_dist=amb_temp,int_method='RK45', d_func = interp.interp1d(
                d_i["MD"], d_i["d_i"], fill_value="extrapolate", kind="previous"
                ),
                 directions=(0,0), friction_factor=1,holdup_factor=1,heat_balance=1,steps=step)


#TECT
    for i in range(0, 10,10):
        zab = schet(i,qu_liq_r=300, wct_r=0.6, p_head_r = (15*101325), t_head_r=293, absep_r = 2.54,
                    md1 = 1400, md2 = 1800, md3 = 3000,
                    tvd1 = 1400, tvd2 = 1800, tvd3=2400,
                     gamma_gas = 0.7,gamma_wat = 1, gamma_oil=0.8, pb = (50 * 101325), t_res = 303.15,
                  rsb = 50, muob = 0.5, bob = 1.5, d_o_1 = 142, d_o_2 =142 , d_o_3 = 142, d_i_1 = 73, d_i_2 = 73,
                  d_i_3 = 73,)
        print('Забойное давлении:',zab, 'атм. при ГФ =',i, 'м3/м3')
        # print(schet_pipe(i,qu_liq_r=300, wct_r=0.6, p_head_r = (15*101325), t_head_r=293, absep_r = 2.54,
        #          md1 = 1400, md2 = 1800, md3 = 2400,
        #           tvd1 = 1400,tvd2 = 1800, tvd3=2400,
        #            gamma_gas = 0.7,gamma_wat = 1, gamma_oil=0.8, pb = (50 * 101325), t_res = 303.15,
        #           rsb = 50, muob = 0.5, bob = 1.5, d_o_1 = 142, d_o_2 =142 , d_o_3 = 142, d_i_1 = 73, d_i_2 = 73,
        #           d_i_3 = 73,))
