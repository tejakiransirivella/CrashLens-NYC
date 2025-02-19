import matplotlib.pyplot as plt
import numpy as np

class Visualization:

    def draw_histogram(self,labels,y1,y2,title,xlabel,ylabel,xlegend,ylegend):
        '''
        Draws a histogram with two sets of data side by side for comparison.
        '''
        label_index = np.arange(len(labels))

        bar_width = 0.35

        plt.bar(label_index - bar_width/2, y1, bar_width, label=xlegend)
        plt.bar(label_index + bar_width/2, y2, bar_width, label=ylegend)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(label_index, labels, rotation=45)  # Set month names as x-ticks with rotation

        plt.legend()

        plt.tight_layout()
        plt.show() 

    def draw_single_histogram(self,x_data,y_data,xlabel,ylabel,title,legend):
        '''
        Draws a single histogram.
        '''
        plt.bar(x_data,y_data,label=legend)

       
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45) 
    
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_line_graph(self,x_data,y_data,xlabel,ylabel,title, max_accidents):
        '''
        Plots a line graph with annotations for the maximum value.
        '''
      
        plt.figure(figsize=(10, 6))  
        plt.plot(x_data,y_data,color='red', marker='x', linestyle='-',linewidth = 1)

       
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.annotate(f"({max_accidents.iloc[0]['CRASH DATE']}, {max_accidents.iloc[0]['rolling_accidents']})", xy=(max_accidents['CRASH DATE'], max_accidents['rolling_accidents']), xytext=(max_accidents['CRASH DATE'], max_accidents['rolling_accidents']+50), 
                    arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=7)

       
        plt.xticks(rotation=45)

       
        plt.grid(True)


        plt.tight_layout() 
        plt.show()
    
    def draw_pie_chart(self,data,labels,title):
        '''
        Draws a pie chart.
        '''
        plt.figure(figsize=(10, 8))
        plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140,textprops={'fontsize': 4})
        plt.title(title )
        plt.axis('equal') 
        plt.show()