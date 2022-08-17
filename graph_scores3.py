# Begin main routine.

##Get list of complex names
from os import walk
import os
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

delta_sasa = np.load("delta_sasa_complex_minus_rl.npy")
scores = np.load("scores.npy")

# plt.scatter(scores, delta_sasa, 50, c="g", alpha=0.5, marker='+',
#             label="SC vs DeltaSASA")
# plt.xlabel("SC_score")
# plt.ylabel("Delta_SASA")
# plt.legend(loc='upper left')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress




x = scores
y = delta_sasa

indicies = np.where(y == 0)

indicies = indicies[::-1]

for i in indicies:
    x = np.delete(x, i)
    y = np.delete(y, i)

indicies = np.where(x == 0)
indicies = indicies[::-1] 

for i in indicies:
    x = np.delete(x, i)
    y = np.delete(y, i)

reg = linregress(x,y)

coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef) 
# poly1d_fn is now a function which takes in x and returns an estimate for y
fig, ax = plt.subplots()
m,b = np.polyfit(x, y, 1)
ax.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker
ax.annotate("r-value = {:.3f}".format(reg[2]**2), (0, 1))

ax.set_title('R-value: ' + str(reg[2]))
plt.show()
# plt.xlim(0, 5)
# plt.ylim(0, 12)

