import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("results_BITSC2.csv")
sns.set(font="Helvetica",font_scale=2.0)
g = sns.catplot(x="rho", y="BD",hue="method", col="nodes",row="muts", data=df, kind="box",palette="Set1",showmeans=True,meanprops={"markersize":10,"marker":"X"},
        margin_titles=False,legend=False,fliersize=2) # , height=4, aspect=.7
g.set_axis_labels("Variance in region coverage","Bourque Distance")
for AX in g.axes.flatten():
    for i,artist in enumerate(AX.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        print(col)
        artist.set_edgecolor(col)
        artist.set_alpha(0.5)
        #artist.set_facecolor('None')
        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*7,i*7+7):
            line = AX.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

for AX in g.axes[2]:
    t_labels = [x.get_text() for x in AX.get_xticklabels()]
    print(t_labels)
    def format_func(value, tick_number):
        if not value in range(len(t_labels)):
            return value
        label = float(t_labels[value])
        if label==0.5:
            return r"$\bf{"+ "{:.1f}".format(label)+r"}$"
        else:
            return "{:.1f}".format(label)
    AX.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
g.axes[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30),
          ncol=2, fancybox=True, shadow=False)
plt.show()

plt.savefig("Comparison_BITSC2.png",dpi=500,bbox_inches="tight",pad_inches=0.1)