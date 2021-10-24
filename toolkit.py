from sixgill.pipesim import Model, update_well_context
from sixgill.definitions import ModelComponents, Parameters, Constants, SystemVariables, ProfileVariables
import pandas as pd
import os.path
from unifloc.tools.units_converter import _m3m3_2_scfstb, __scfstb2m3m3

a1 = [i for i in range(0,200, 10)]
b1 = []
res10 = []
for i in a1:
    c1 = _m3m3_2_scfstb(i)
    b1.append(c1)

model = Model.open(r'C:/Users/123/Desktop/rab/MultiphaseFlow/calc/MultiphaseFlow/test.pips')
model.set_value(context= 'BOFluid 2', parameter = "WaterCut", value = 15)
for i in b1:
    model.set_value(context= 'BOFluid 2', parameter = 'GOR', value = i) #scf/stb

    system_variables = [
            SystemVariables.PRESSURE,
           
    ]

    parameters = {
    Parameters.PTProfileSimulation.OUTLETPRESSURE:220,  #psia
    Parameters.PTProfileSimulation.LIQUIDFLOWRATE:3584,  #stb/d
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
        res10.append(res)
        print (c/14.699)


model.save()
model.close()

print(a1,res10)
df1 = pd.DataFrame({'GOR': a1,
                    'p down': res10})
df1.to_excel('./wct15570.xlsx')


