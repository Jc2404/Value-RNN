import matplotlib
from matplotlib.backends.qt_compat import QT_API
print("backend:", matplotlib.get_backend())
print("QT_API:", QT_API)

import tkinter
print("tkinter ok")

import PyQt6
print("PyQt5 ok")

import matplotlib.pyplot as plt
import os
out_path = os.path.abspath("test_plot.png")
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9], marker="o")
plt.title("save test")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print("saved to:", out_path)