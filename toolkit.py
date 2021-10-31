from sixgill.pipesim import Model, update_well_context
from sixgill.definitions import ModelComponents, Parameters, Constants, SystemVariables, ProfileVariables
import pandas as pd
import os.path
from unifloc.tools.units_converter import _m3m3_2_scfstb, __scfstb2m3m3

a1 = [i for i in range(0,1000, 25)]
b1 = []
res10 = []
res25 = []
res40 = []
res60 = []
for i in a1:
    c1 = _m3m3_2_scfstb(i)
    b1.append(c1)

model = Model.open(r'C:/Users/123/Desktop/rab/MultiphaseFlow/calc/MultiphaseFlow/test34500.pips')


# model.set_value(context= 'BOFluid', parameter = "WaterCut", value = 10)
# for i in b1:
#     model.set_value(context= 'BOFluid', parameter = 'GOR', value = i) #scf/stb
#     system_variables = [
#             SystemVariables.PRESSURE,
#     ]
#     parameters = {
#     Parameters.PTProfileSimulation.OUTLETPRESSURE:220,  #psia
#     Parameters.PTProfileSimulation.LIQUIDFLOWRATE:5660.83,  #stb/d
#     Parameters.PTProfileSimulation.FLOWRATETYPE:Constants.FlowRateType.LIQUIDFLOWRATE,
#     Parameters.PTProfileSimulation.CALCULATEDVARIABLE:Constants.CalculatedVariable.INLETPRESSURE,
#     }
#     results = model.tasks.ptprofilesimulation.run(producer="Well",
#                                parameters=parameters,
#                                system_variables=system_variables)
#     for case, node_res in results.node.items():
#         a=list(node_res.items())[-1]
#         b=list(a)[-1]
#         c=list(b.values())[-1]
#         res = c/14.699
#         res10.append(res)
#         print (c/14.699)


# model.set_value(context= 'BOFluid', parameter = "WaterCut", value = 25)
# for i in b1:
#     model.set_value(context= 'BOFluid', parameter = 'GOR', value = i) #scf/stb
#     system_variables = [
#             SystemVariables.PRESSURE,
#     ]
#     parameters = {
#     Parameters.PTProfileSimulation.OUTLETPRESSURE:220,  #psia
#     Parameters.PTProfileSimulation.LIQUIDFLOWRATE:5660.83,  #stb/d
#     Parameters.PTProfileSimulation.FLOWRATETYPE:Constants.FlowRateType.LIQUIDFLOWRATE,
#     Parameters.PTProfileSimulation.CALCULATEDVARIABLE:Constants.CalculatedVariable.INLETPRESSURE,
#     }
#     results = model.tasks.ptprofilesimulation.run(producer="Well",
#                                parameters=parameters,
#                                system_variables=system_variables)
#     for case, node_res in results.node.items():
#         a=list(node_res.items())[-1]
#         b=list(a)[-1]
#         c=list(b.values())[-1]
#         res = c/14.699
#         res25.append(res)
#         print (c/14.699)


# model.set_value(context= 'BOFluid', parameter = "WaterCut", value =40)
# for i in b1:
#     model.set_value(context= 'BOFluid', parameter = 'GOR', value = i) #scf/stb
#     system_variables = [
#             SystemVariables.PRESSURE,
#     ]
#     parameters = {
#     Parameters.PTProfileSimulation.OUTLETPRESSURE:220,  #psia
#     Parameters.PTProfileSimulation.LIQUIDFLOWRATE:5660.83,  #stb/d
#     Parameters.PTProfileSimulation.FLOWRATETYPE:Constants.FlowRateType.LIQUIDFLOWRATE,
#     Parameters.PTProfileSimulation.CALCULATEDVARIABLE:Constants.CalculatedVariable.INLETPRESSURE,
#     }
#     results = model.tasks.ptprofilesimulation.run(producer="Well",
#                                parameters=parameters,
#                                system_variables=system_variables)
#     for case, node_res in results.node.items():
#         a=list(node_res.items())[-1]
#         b=list(a)[-1]
#         c=list(b.values())[-1]
#         res = c/14.699
#         res40.append(res)
#         print (c/14.699)

model.set_value(context= 'BOFluid', parameter = "WaterCut", value = 60)
for i in b1:
    model.set_value(context= 'BOFluid', parameter = 'GOR', value = i) #scf/stb
    system_variables = [
            SystemVariables.PRESSURE,
    ]
    parameters = {
    Parameters.PTProfileSimulation.OUTLETPRESSURE:220,  #psia
    Parameters.PTProfileSimulation.LIQUIDFLOWRATE:5660.83,  #stb/d
    Parameters.PTProfileSimulation.FLOWRATETYPE:Constants.FlowRateType.LIQUIDFLOWRATE,
    Parameters.PTProfileSimulation.CALCULATEDVARIABLE:Constants.CalculatedVariable.INLETPRESSURE,
    }
    results = model.tasks.ptprofilesimulation.run(producer="Well",
                               parameters=parameters,
                               system_variables=system_variables)
    for case, node_res in results.node.items():
        a=list(node_res.items())[-1]
        b=list(a)[-1]
        c=list(b.values())[-1]
        res = c/14.699
        res60.append(res)
        print (c/14.699)


model.save()
model.close()


# df1 = pd.DataFrame({'GOR': a1,
#                     'p down': res10})
# df1.to_excel('./t3q900wct0.1.xlsx')

# df2 = pd.DataFrame({'GOR': a1,
#                     'p down': res25})
# df2.to_excel('./t3q900wct0.25.xlsx')

# df3 = pd.DataFrame({'GOR': a1,
#                     'p down': res40})
# df3.to_excel('./t3q900wct0.4.xlsx')

df4 = pd.DataFrame({'GOR': a1,
                    'p down': res60})
df4.to_excel('./t3q900wct0.6.xlsx')




