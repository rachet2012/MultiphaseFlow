import warnings
import math as m
import numpy as np
import pandas as pd
import scipy.constants as CONST
import scipy.optimize as sp

import unifloc.pipe._friction as fr
import unifloc.pipe._hydrcorr as hr

warnings.filterwarnings("ignore",category=RuntimeWarning)


#TODO fsolve(экспоненты и логарифм, пока подходит только fsolve)
class HasanKabirAnn(hr.HydrCorr):
    """
    Класс для расчета градиента давления в затрубном пространстве по корреляции HasanKabir/CaetanoBrill
    Определяются структура потока, истинная концентрация газа, плотность смеси,
    градиенты на гравитацию, трение, ускорение.

    """

    def __init__(self,   d: float = 73, d_o_m: float = 142,
         ) -> None:
        """
        :param d: внешний диаметр НКТ, мм
        :param d_o_m: внутренний диаметр ЭК, мм
        """
        self.d = d / 1000
        self.d_o_m = d_o_m / 1000


    def calc_params(self):
        """
        Метод расчета дополнительных параметров, необходимых для расчета градиента давления в трубе
        по методике Hasan Kabir

        :param f_m2: площадь сечения затрубного пространства, м2
        :param vs_gas_msec: приведенная скорость газа, м/с
        :param vs_liq_msec: приведенная скорость жидкости, м/с
        :param v_mix_msec: приведенная скорость смеси, м/с
        :param _d_equ_m: эквивалентный диаметр затрубного пространства, м2
        """
        f_m2 =  CONST.pi * ((self.d_o_m/2)**2 - (self.d/2)**2)
        self.vsg = self.qg / f_m2
        self.vsl = self.ql / f_m2
        self.vsm = (self.qg + self.ql) / f_m2
        self._d_equ_m = self.d_o_m - self.d


    def _mixture_velocity_Caetano(self, initial_f):
        """
        Метод для определения критической скорости dispersed bubbly flow
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        """
        rho_m_rash_kgm3 = (self.qg / (self.qg + self.ql)
                                * self.rg+ (1 - self.qg / (self.qg +
                                self.ql)) * self.rl)
        friction_coeff = self._actual_friction_coef(rho_m_rash_kgm3)
        right_part = 2 * (initial_f ** 1.2) * (friction_coeff** 0.4) * ((2 / self._d_equ_m) **
                     0.4) * ((self.rl/ self.stlg) ** 0.6 ) *(0.4 * self.stlg / (
                         (self.rl- self.rg) * CONST.g) ** 0.5)
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

    def _calc_fp(self):
        """
        Метод для расчета критических значений скоростей
        TWO-PHASE FLOW IN VERTICAL AND INCLINED ANNULI
        Upward Vertical Two-Phase Flow Through an Annulus—Part I for to dispersed
        """
        #bubble to slug transition [4]
        v_d_msec = 1.53 * (CONST.g * self.stlg * (self.rl
                        - self.rg) / (self.rl)**2 ) ** 0.25

        self.vs_gas_bubble2slug_msec = ((1.2 * self.vsl + v_d_msec) / (4
                                    - 1.2)) * np.sin(self.angle * np.pi/180)
        #to annular transirion [17]
        self.vs_gas_2annular_msec = ((3.1 * (self.stlg * CONST.g * (self.rl-
                                    self.rg) / (self.rg) ** 2) ** 0.25 + 1)
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
        k_ratio_d = self.d / self.d_o_m
        frict = fr.Friction(self._d_equ_m)
        mu_mix_pasec = (self.vsl / self.vsm * self.mul
                            + self.vsg / self.vsm * self.mug)
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
        v_d_msec = (1.53 * (CONST.g * self.stlg * (self.rl- self.rg)
                     / (self.rl)**2 ) ** 0.25) #3
        v_gas_msec = 1.2 * self.vsm + v_d_msec #1
        self.epsi = self.vsg / v_gas_msec #2
        self.hl = 1 - self.epsi

    def _calc_slug_churn(self) -> float:
        """
        Метод для расчета истинной объемной концентрации газа в slug и churn flow
        """
        self.v_dt_msec = (1.2 *(self.vsg + self.vsl) + 0.345 *
                        (CONST.g * (self.d + self.d_o_m)) ** 0.5
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
        self.v_lls = ((self.vsl + self.vsg) - 1.53 * ((self.rl- self.rg) #23
                    * CONST.g * self.stlg / (self.rl**2)) ** 0.25 * self.h_ls ** 0.5 * (1 - self.h_ls))
        self.v_gls = ((1.53 * ((self.rl- self.rg) * CONST.g * self.stlg / (self.rl**2) #20
                        ) ** 0.25 * self.h_ls ** 0.5) + self.v_lls)
        act_len_lf = (self._actual_film_length())
        v_llf = m.fabs((CONST.g * 2 * act_len_lf) ** 0.5 - self.v_dt_msec) #47
        grad_p_acc = (self.rl* (h_lf / len_su) * (v_llf - self.v_dt_msec)
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
        v_dt_msec = (0.345 + 0.1 * (self.d / self.d_o_m)) *((CONST.g * self.d_o_m * (
                        self.rl- self.rg)/(self.rg)) ** 0.5)
        len_su = 1
        act_len_lf = len_su
        v_llf = (CONST.g * 2 * act_len_lf) ** 0.5 - v_dt_msec
        grad_p_acc_an = (self.rl* (self.hl / len_su) * (v_llf - v_dt_msec)
                    * v_llf)
        return grad_p_acc_an

    def _calc_hl(self):
        """
        Метод для вычисления концентрации газа в потоке кольцевой структуры[77]
        Допустил, что толщина пленки жидкости на внешней(или внутренней) трубе известна
        """
        delta_o = 0.005
        k_ratio_d = self.d / self.d_o_m
        angle_wt_average = (1 / (1 - k_ratio_d ** 2) * (2 * m.asin(k_ratio_d) + 2 * #[88]
                             k_ratio_d  * (1 - k_ratio_d ** 2) ** 0.5 - CONST.pi *
                             k_ratio_d ** 2))
        t_ratio = angle_wt_average / ((2 * CONST.pi - angle_wt_average) * k_ratio_d)
        delta_i = delta_o * t_ratio
        phi = (10 ** 4 * self.vsg * self.mug / self.stlg * (self.rg#[79]
                    / self.rl) ** 0.5)
        fe = 1 - m.exp((phi - 1.5) * (-0.125))
        self.hl = (4 / ((self.d_o_m) * (1 - k_ratio_d ** 2)) * (delta_o * (1 - delta_o / self.d_o_m)#[77]
                        + delta_i * k_ratio_d * (1 + delta_i / self.d) + self.vsl * fe
                        / ((self.vsl * fe + self.vsg) * (1 - k_ratio_d ** 2)) *
                        (1 - k_ratio_d ** 2 - 4 * delta_o / self.d_o_m * (1 - delta_o / self.d_o_m)
                        - 4 * delta_i * k_ratio_d / self.d_o_m * (1 + delta_i / self.d))))
        self.epsi = 1 - self.hl

    def calc_rho_mix(self):
        """
        Метод для расчета плотности смеси
        """
        self._calc_fp()
        if self.flow_pattern == 0 or self.flow_pattern == 1:
            self._calc_bubbly()
        elif self.flow_pattern == 2 or self.flow_pattern == 3:
            self._calc_slug_churn()
        elif self.flow_pattern == 4:
            self._calc_hl()
        self.rho_mix_kgm3 = self.rl * (1 - self.epsi) + self.rg* self.epsi

    def calc_grav(self):
        if self.flow_pattern == 0 or self.flow_pattern == 1 or self.flow_pattern == 4: #[5-14]
            self.dp_dl_gr = self.rho_mix_kgm3 * CONST.g * np.sin(self.angle * np.pi/180)
        elif self.flow_pattern == 2 or self.flow_pattern == 3 :
            self.rho_slug_kgm3 = self.rg* self.epsi_s + self.rl* (1 - self.epsi_s) # В соответствии c [51]
            self.dp_dl_gr = self.rho_slug_kgm3 * CONST.g  * self.len_s_m *  np.sin(self.angle * np.pi/180)
        return self.dp_dl_gr

    def calc_fric(self):
        if self.flow_pattern == 0 or self.flow_pattern == 1 or self.flow_pattern == 4: #[5-14]
            friction_coeff_s = self._actual_friction_coef(self.rho_mix_kgm3)
            self.dp_dl_fr = ((4 * friction_coeff_s / (self._d_equ_m )
                                     * self.vsm ** 2 / 2) * self.rho_mix_kgm3)
        elif self.flow_pattern == 2 or self.flow_pattern == 3 :
            self.rho_slug_kgm3 = self.rg* self.epsi_s + self.rl* (1 - self.epsi_s) # В соответствии c [51]
            friction_coeff_s = self._actual_friction_coef(self.rho_slug_kgm3)
            self.dp_dl_fr = ((2 * friction_coeff_s / self._d_equ_m * self.rho_slug_kgm3)#[53]
                                     * (self.vsg + self.vsl) **2 * self.len_s_m)
        return self.dp_dl_fr

    def calc_acc(self):
        if self.flow_pattern == 0 or self.flow_pattern == 1: #[5-14]
            self.dp_dl_acc = 0
        elif self.flow_pattern == 2 or self.flow_pattern == 3 :
            self.rho_slug_kgm3 = self.rg* self.epsi_s + self.rl* (1 - self.epsi_s) # В соответствии c [51]
            self.dp_dl_acc = self._acceler_grad_p()
        elif self.flow_pattern == 4: #методику не приводят
            self.dp_dl_acc = self._acceler_grad_p_annular()
        return self.dp_dl_acc

    def calc_grad(self,  theta_deg,
            eps_m,
            ql_rc_m3day,
            qg_rc_m3day,
            mul_rc_cp,
            mug_rc_cp,
            sigma_l_nm,
            rho_lrc_kgm3,
            rho_grc_kgm3,
            c_calibr_grav,
            c_calibr_fric,
            h_mes,flow_direction,
            vgas_prev,
            rho_gas_prev,
            h_mes_prev,
            calc_acc,
            rho_mix_rc_kgm3):
        """
        Метод для расчета градиента давления
        Upward Vertical Two-Phase Flow Through an Annulus—Part II

        :return: суммарный градиент давления, Па/м

        """

        self.angle = theta_deg
        self.abseps = eps_m / 100000
        self.ql = ql_rc_m3day
        self.qg = qg_rc_m3day
        self.stlg = sigma_l_nm
        self.rl = rho_lrc_kgm3
        self.rg= rho_grc_kgm3
        self.mul = mul_rc_cp
        self.mug = mug_rc_cp

        self.calc_params()
        self.calc_rho_mix()

        self.dp_dl_gr = self.calc_grav()
        self.dp_dl_fr = self.calc_fric()
        self.dp_dl_acc = self.calc_acc()

        self.dp_dl = self.dp_dl_fr  + self.dp_dl_gr + self.dp_dl_acc
        return self.dp_dl

