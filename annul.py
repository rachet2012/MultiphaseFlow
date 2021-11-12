from typing import Union

import numpy as np
import unifloc.pipe._pipe as pip
import unifloc.pvt.fluid_flow as  fl
import pandas as pd
import unifloc.common.trajectory as tr
import unifloc.pipe._friction as fr

import scipy.integrate as integr
import unifloc.common.ambient_temperature_distribution as amb


class Annul(pip.Pipe):
    def __init__(
        self,
        fluid: fl.FluidFlow,
        d_casing: Union[float, pd.DataFrame, dict],
        d_tubing: Union[float, pd.DataFrame, dict],
        roughness: float,
        hydr_corr_type="HasanKabir"
    ):
        self.fluid = fluid
        self.d_casing = d_casing
        self.d_tubing = d_tubing
        self.roughness = roughness
        self.hydr_corr_type = hydr_corr_type

    def __define_hvrho(self, h, calc_direction):
        """
        Функция для определения предыдущей успешной глубины, скорости и плотности на этой глубине

        :param h: текущая глубина, м
        :param calc_direction: направление расчета
        :return: предыдущая успешная глубина, м;
        :return: скорость газа на этой глубине, м/c;
        :return: плотность газа на этой глубине, кг/м3
        """
        if (calc_direction < 0 and h > self._h_prev_next) or (
            calc_direction > 0 and h < self._h_prev_next
        ):
            h_prev_act = self._h_prev_next
            rho_n_prev_act = self._rho_n_prev_next
            vsm_prev_act = self._vsm_prev_next

            self._h_prev = self._h_prev_next
            self._vsm_prev = self._vsm_prev_next
            self._rho_n_prev = self._rho_n_prev_next

            self._h_prev_hist = np.append(self._h_prev_hist, self._h_prev)
            self._vsm_prev_hist = np.append(self._vsm_prev_hist, self._vsm_prev)
            self._rho_n_prev_hist = np.append(self._rho_n_prev_hist, self._rho_n_prev)

        elif (calc_direction < 0 and h >= self._h_prev) or (
            calc_direction > 0 and h <= self._h_prev
        ):
            h_prev_act = self._h_prev
            rho_n_prev_act = self._rho_n_prev
            vsm_prev_act = self._vsm_prev
        else:
            if calc_direction > 0:
                mask = self._h_prev_hist >= h
            else:
                mask = self._h_prev_hist <= h

            self._h_prev_hist = self._h_prev_hist[mask]
            self._h_prev = self._h_prev_hist[-1]
            h_prev_act = self._h_prev

            self._rho_n_prev_hist = self._rho_n_prev_hist[mask]
            self._rho_n_prev = self._rho_n_prev_hist[-1]
            rho_n_prev_act = self._rho_n_prev

            self._vsm_prev_hist = self._vsm_prev_hist[mask]
            self._vsm_prev = self._vsm_prev_hist[-1]
            vsm_prev_act = self._vsm_prev

        return h_prev_act, vsm_prev_act, rho_n_prev_act


    def __integr_func(
        self,
        h,
        pt,
        trajectory,
        amb_temp_dist,
        directions,
        d_i_func,
        d_o_func
    ):
        """
        Функция для интегрирования трубы

        :param h: текущая глубина, м
        :param pt: текущее давление, Па и текущая температура, К
        :param trajectory: объект с траекторией
        :param amb_temp_dist: объект с распределением температуры породы
        :param directions: множитель для направления потока, флаг направления расчета
        :param holdup_factor: коэффициент истинного содержания жидкости/гравитации
        :param friction_factor: коэффициент трения
        :param d_func: объект расчет диаметра по глубине скважины
        :param heat_balance: опция учета теплопотерь

        Parameters
        ----------

        :return: градиент давления в заданной точке трубы
        при заданных термобарических условиях, Па/м
        :return: градиент температуры в заданной точке трубы
        при заданных термобарических условиях, К/м
        """

        # Условие прекращения интегрирования
        if np.isnan(pt[0]) or pt[0] <= 0:
            return False

        p, t = pt

        # Пересчет PVT свойств на заданной глубине
        self.fluid.calc_flow(p, t)

        # Определение текущей величины шага по сравнению
        # с последним успешным, сохранение скорости и плотности

        h_prev, vgas_prev, rho_gas_prev = self.__define_hvrho(h, directions[1])

        # Вычисление угла
        theta_deg = trajectory.calc_angle(h_prev, h)

        # Вычисление диаметра


        # Расчет градиента давления, используя необходимую гидравлическую корреляцию
        dp_dl = directions[0] * self.hydrcorr.calc_grad(theta_deg, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,)

        # Расчет геотермического градиента
        dt_dl = amb_temp_dist.calc_geotemp_grad(h)

        # Сохранение предыдущего значения для расчета угла на следующем шаге
        self._h_prev_next = h
        self._rho_n_prev_next = self.fluid.rg
        self._vsm_prev_next = self.hydrcorr.vsg

        return dp_dl, dt_dl

    def integrate_pipe(
        self,
        p0,
        t0,
        h0,
        h1,
        trajectory,
        amb_temp_dist,
        int_method,
        directions,
        steps,
    ):
        """
        Метод для интегрирования давления, температуры в трубе

        Parameters
        ----------
        :param p0: начальное давление, Па
        :param t0: начальная температура, К
        :param h0: начальная глубина, м
        :param h1: граничная глубина, м
        :param trajectory: объект с траекторией
        :param amb_temp_dist: объект с распределением температуры породы
        :param int_method: метод интегрирования
        :param directions: множитель для направления потока, флаг направления расчета
        :param friction_factor: коэффициент поправки на трение
        :param holdup_factor: коэффициент поправки на истинное содержание жидкости
        :param steps: массив узлов для которых необходимо расcчитать давление
        :param d_func: диаметр, функция или число, м
        :param heat_balance: опция учета теплопотерь


        Returns
        -------
        :return: массив глубин, м и давлений, Па

        """
        # Задание нулевых значений глубины, скорости и плотности
        self._h_prev = h0
        self._h_prev_next = h0
        self._h_prev_hist = np.array([h0])
        self._vsm_prev = 0
        self._vsm_prev_next = 0
        self._vsm_prev_hist = np.array([0])
        self._rho_n_prev = 0
        self._rho_n_prev_next = 0
        self._rho_n_prev_hist = np.array([0])


        dptdl_integration = integr.solve_ivp(
            self.__integr_func,
            t_span=(h0, h1),
            y0=[p0, t0],
            method=int_method,
            args=(
                trajectory,
                amb_temp_dist,
                directions,
                self.d_casing,
                self.d_tubing,
            ),
            t_eval=steps,
        )
        return dptdl_integration


if __name__ == '__main__':
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
        pvt = fl.FluidFlow(qu_liq_r/86400, wct_r, pvt_model_data)
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
        d_o = 142
        d_i = 73
        ambient_temperature_data = {"MD": [0, md3], "T": [t_head_r, t_res]}
        amb_temp = amb.AmbientTemperatureDistribution(ambient_temperature_data)
        step = [i for i in range(0, tvd3+50, 50)]
        pip = Annul(fluid=pvt,d_casing=d_o, d_tubing=d_i, roughness=absep_r,hydr_corr_type='HasanKabir')
        return pip.integrate_pipe(p0 = p_head_r,t0= t_head_r,h0=0,h1=tvd3,trajectory= trajector,
                 amb_temp_dist=amb_temp,int_method='RK45',
                 directions=(1,0),steps=step)

    print(schet_pipe(0,qu_liq_r=300, wct_r=0.6, p_head_r = (15*101325),
                 t_head_r=293, absep_r = 2.54,
                 md1 = 1400, md2 = 1800, md3 = 3000,
                  tvd1 = 1400,tvd2 = 1800, tvd3=3000,
                   gamma_gas = 0.7,gamma_wat = 1, gamma_oil=0.8,
                   pb = (50 * 101325), t_res = 363.15,
                  rsb = 50, muob = 0.5, bob = 1.5,
                  d_o_1 = 142, d_o_2 =142 , d_o_3 = 142,
                  d_i_1 = 73, d_i_2 = 73, d_i_3 = 73,))