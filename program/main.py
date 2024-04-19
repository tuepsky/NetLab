'''
NetLab
Main program
Martin Reiche 2024
'''

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import Global as g
from SetTrainTest import SetTrainTest
from Inspection import Inspection

class Gui(tk.Tk):
    def __init__(self):
        super(Gui, self).__init__()
        self.title('NetLab 0.4 -  martin-reiche.de 2024')
        self.minsize(770, 800)

        # Tabulator
        customed_style = ttk.Style()
        customed_style.theme_use("default")
        customed_style.configure('TNotebook.Tab', padding=[6, 6], font=g.fontLabel)
        customed_style.map('TNotebook.Tab', background=[('selected', g.bgDark)])
        customed_style.configure('TButton', foreground='white')

        tab_control = ttk.Notebook(self, style='Custom.TNotebook')
        self.tabSetTrainTest = SetTrainTest(tab_control)
        tab_control.add(self.tabSetTrainTest, text='Setup, Train & Test')
        self.tabInspection = Inspection(tab_control)
        tab_control.add(self.tabInspection, text='Inspect')
        tab_control.pack(expand=1, fill='both')
        tab_control.bind('<<NotebookTabChanged>>', self.tab_changed)
        
        # Status bar
        g.test_file_name = tk.StringVar()
        status_bar_1 = tk.Label(self, textvariable=g.test_file_name, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                font=("Arial", 10), pady=4, background=g.bgDark)
        status_bar_1.pack(side=tk.BOTTOM, fill=tk.X)

        g.training_file_name = tk.StringVar()
        status_bar_2 = tk.Label(self, textvariable=g.training_file_name, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                font=("Arial", 10), pady=4, background=g.bgDark)
        status_bar_2.pack(side=tk.BOTTOM, fill=tk.X)

        self.current_status = tk.StringVar()
        self.status_bar_3 = tk.Label(self, textvariable=self.current_status, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                font=("Arial", 12), pady=4, background=g.bgDark, fg='yellow')
        self.status_bar_3.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, text):
        self.current_status.set(text)
        self.status_bar_3.update()

    def tab_changed(self,event):
        if event.widget.index('current') == 1: # inspection tab
            self.tabInspection.create_inspection_display()
            self.tabInspection.apply_pattern(0)


g.gui = Gui() # The gui is a global instance!
g.gui.mainloop()

