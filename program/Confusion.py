# Shows the confusion matrix in a separate window
import tkinter as tk
import Global as g

class ConfusionMatrix:
    
    def __init__(self, parent, labels, matrix):
        rows = len(labels) + 1
        cell_w = 50
        cell_h = 30
        margin_x = 60
        margin_y = 40
        width = margin_x + rows * cell_w + 10
        height = margin_y + rows * cell_h + 10
        top = tk.Toplevel(parent)
        top.geometry("%dx%d%+d%+d" % (width, height, 200, 200))
        top.title('Confusion Matrix')
        
        canvas = tk.Canvas(top, width=width, height=height, bg=g.bgLight, highlightthickness=0)
        canvas.place(x=0, y=0)
        ## Draw legend
        lbl = tk.Label(top, text='detected label', font=g.fontLabel, bg=g.bgLight)
        lbl.place(x = margin_x + cell_w + 10, y = 10)

        lbl = tk.Label(top, text='actual\nlabel', font=g.fontLabel, bg=g.bgLight)
        lbl.place(x=6, y=80)
        
        ## Draw frame
        # top
        start_x = margin_x + cell_w
        canvas.create_line((start_x, margin_y,
                            start_x + (rows-1) * cell_w, margin_y), width=2)
        # left
        start_y = margin_y + cell_h
        canvas.create_line((margin_x, start_y,
                            margin_x, start_y + (rows-1) * cell_h), width=2)
        # bottom
        canvas.create_line((margin_x, margin_y + rows * cell_h,
            margin_x + rows * cell_w, margin_y + rows * cell_h), width=2)
        # right
        canvas.create_line((margin_x + rows * cell_w, margin_y,
            margin_x + rows * cell_w, margin_y + rows * cell_h), width=2)
        
        # write labels in top row and left column
        del_top = 4   # distance of labels to upper line
        del_left = 4  # distance of labels to left line
        
        for i, label in enumerate(labels):
            lbl = tk.Label(top, text=label, font=g.fontLabel, bg=g.bgLight)
            lbl.place(x = margin_x + (i+1) * cell_w + del_left, y = margin_y + del_top)
            lbl = tk.Label(top, text=label, font=g.fontLabel, bg=g.bgLight)
            lbl.place(x = margin_x + del_left, y = margin_y + (i+1) * cell_h + del_top)
            canvas.create_line((margin_x, margin_y + (i+1) * cell_h,
                margin_x + rows * cell_w, margin_y + (i+1) * cell_h), width=2)            
            canvas.create_line((margin_x + (i+1) * cell_w, margin_y,
                margin_x + (i+1) * cell_w, margin_y + rows * cell_h), width=2)
            
        # fill matrix cells
        for r, row in enumerate(matrix): # fill matrix cells
            for c, val in enumerate(row):
                if val:
                    text = int(val)
                else:
                    text = ''
                lbl = tk.Label(top, text=text, font=g.fontLabel, bg=g.bgLight)
                lbl.place(x = margin_x + (c+1) * cell_w + del_left,
                          y = margin_y + (r+1) * cell_h + del_top)
                
