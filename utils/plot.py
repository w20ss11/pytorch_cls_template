import os
import numpy as np
import re
import pdb
import  matplotlib.pyplot as plt

log_dir = "0602_223507"
var = "Loss"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(BASE_DIR, "save", log_dir, "log.txt")
lines = open(log_path, "r").readlines()
vars = []
for line in lines:
    var_search = re.search( r'.*'+var+':(\d+\.+\d+)', line, re.M|re.I)
    if var_search:
        vars.append(float(var_search.group(1)))
time = np.arange(len(vars))
print(time)
print(vars)

plt.figure()
plt.xlim((np.min(time)-1, np.max(time)+1))#设置x轴范围
plt.ylim((np.min(vars)-1, np.max(vars)+1))#设置轴y范围
# plt.plot(time, vars, color='red', linewidth=1.0, linestyle='-')
plt.plot(time, vars, '-o')
#设置坐标轴含义， 注：英文直接写，中文需要后面加上fontproperties属性
plt.xlabel(u'time',fontproperties='SimHei')
plt.ylabel(u'损失loss',fontproperties='SimHei')
plt.show()