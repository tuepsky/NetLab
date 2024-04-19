import tkinter as tk
import numpy as np
from copy import deepcopy
import Global as g

class Inspection(tk.Frame):

    def create_manual_pattern(self):
        self.input_mode = "manual"
        self.button_clear.configure(state=tk.NORMAL)
        self.spin_box_record.configure(state=tk.DISABLED)
        index = int(self.spin_box_record.get()) - 1
        pattern = g.allTestPattern
        if len(pattern) == 0:
            return
        label, data = pattern[index]
        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())
        self.input_layer_excitation = deepcopy(data).reshape(number_rows, number_cols)
        self.show_manual_pattern()

    def show_manual_pattern(self):
        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())
        for row in range(number_rows):
            for col in range(number_cols):
                bright = 255 - int(self.input_layer_excitation[row][col] * 255)
                color = "#%02x%02x%02x" % (bright, bright, bright)
                tag = self.input_layer_squares[row][col]
                self.canvas_input.itemconfig(tag, fill=color)

    def clear_manual_pattern(self):
        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())
        self.input_layer_excitation = np.zeros((number_rows, number_cols))
        self.show_manual_pattern()

    def show_input_pattern(self, index):
        self.input_mode = "record"
        self.spin_box_record.configure(state=tk.NORMAL)
        self.button_clear.configure(state=tk.DISABLED)

        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())
        pattern = g.allTestPattern
        if len(pattern) == 0:
            return
        self.spin_box_record.configure(from_=1, to=len(pattern))
        self.current_record.set(index+1)
        pat, data = pattern[index]
        self.pattern.set(g.outputLayerLabels[int(pat)])
        data = data.reshape(number_rows, number_cols)
        for row in range(number_rows):
            for col in range(number_cols):
                bright = 255 - int(data[row][col] * 255)
                color = "#%02x%02x%02x" % (bright, bright, bright)
                tag = self.input_layer_squares[row][col]
                self.canvas_input.itemconfig(tag, fill=color)

    def spin_record(self):
        index = self.spin_box_record.get()
        self.apply_pattern(int(index) - 1)

    def create_inspection_display(self):
        if len(g.allTestPattern) == 0:
            return
        self.create_input_display()
        self.create_hidden_histogram()
        self.create_output_histogram()

    def create_input_display(self):
        # Delete old, if any
        drawn = self.canvas_input.find_all()
        for d in drawn:
            self.canvas_input.delete(d)

        # Create input layer display
        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())

        one_row = [None for _ in range(number_cols)]
        self.input_layer_squares = []
        for _ in range(number_rows):
            self.input_layer_squares.append(one_row[:])

        input_layer_x = 0
        input_layer_y = 0
        input_layer_height = 200

        self.neuron_size = input_layer_height / max(number_rows, number_cols)

        for row in range(number_rows):
            for col in range(number_cols):
                x0 = input_layer_x + self.neuron_size * col
                x1 = x0 + self.neuron_size
                y0 = input_layer_y + self.neuron_size * row
                y1 = y0 + self.neuron_size
                handle = self.canvas_input.create_rectangle(x0, y0, x1, y1,
                                                           fill=g.bgLight,
                                                           outline="#808080")
                self.input_layer_squares[row][col] = handle

    def create_hidden_histogram(self):
        self.clear_hidden_histogram()

    def create_output_histogram(self):
        for label in self.output_labels:
            label.destroy()
        self.output_labels = []
        output_layer_labels = g.outputLayerLabels
        canvas_height = int(self.canvas_output.cget("height"))
        bar_height = min(canvas_height // len(output_layer_labels), 25)
        for i in range(len(output_layer_labels)):
            label = tk.Label(self, text=output_layer_labels[i],
                             font=g.fontLabel, background=g.bgDark,
                             width=8, anchor='e', justify='right')
            label.place(x=self.column_4, y=self.rows[0] + i * bar_height)
            self.output_labels.append(label)
        pass

    def clear_hidden_histogram(self):
        drawn = self.canvas_hidden.find_all()
        for d in drawn:
            self.canvas_hidden.delete(d)
            
    def show_hidden_histogram(self, excitation):
        self.clear_hidden_histogram()
        canvas_width = int(self.canvas_hidden.cget("width"))
        canvas_height = int(self.canvas_hidden.cget("height"))
        bar_height = min(canvas_height // len(excitation), 25)
        for i_bar in range(len(excitation)):
            x0 = 0
            x1 = int(canvas_width * excitation[i_bar]) + 3
            y0 = i_bar * bar_height
            y1 = y0 + bar_height
            self.canvas_hidden.create_rectangle(x0, y0, x1, y1,
                                               fill="black", outline=g.bgDark)

    def show_output_histogram(self, excitation):
        drawn = self.canvas_output.find_all()
        for d in drawn:
            self.canvas_output.delete(d)

        canvas_width = int(self.canvas_output.cget("width"))
        canvas_height = int(self.canvas_output.cget("height"))
        bar_height = min(canvas_height // len(excitation), 25)
        for i_bar in range(len(excitation)):
            x0 = 0
            x1 = int(canvas_width * excitation[i_bar]) + 3
            y0 = i_bar * bar_height
            y1 = y0 + bar_height
            self.canvas_output.create_rectangle(x0, y0, x1, y1,
                                               fill="black", outline=g.bgDark)

    def apply_pattern(self, index):
        if index < 0:
            index = int(self.spin_box_record.get()) - 1
        pattern = g.allTestPattern
        # Check index validity
        if len(pattern) - index < 1:
            return
        # Show the input pattern
        self.show_input_pattern(index)
        # Run the network
        self.output_index, data = pattern[index]
        self.run_network(data)

    def run_network(self, data):
        if not g.neuroNet:
            return
        # Run the neuron network and display layer excitation
        hidden_excitation, output_excitation = g.neuroNet.run(data)
        if hidden_excitation is not None:
            self.show_hidden_histogram(hidden_excitation)
        else:
            self.clear_hidden_histogram()
        self.show_output_histogram(output_excitation)

        # Evaluate detection success
        max_excitation = -100
        for i in range(len(output_excitation)):
            if output_excitation[i] > max_excitation:
                max_excitation = output_excitation[i]
                i_max = i

        self.detected.set(g.outputLayerLabels[i_max])
        if int(self.output_index) == i_max:
            self.lbl_value_detected.configure(background=g.bgLight)
        else:
            self.lbl_value_detected.configure(background="Yellow")

        self.expected.set(g.outputLayerLabels[int(self.output_index)])

    def spin_box_record_changed(self, _):
        val = self.spin_box_record.get()
        self.current_record.set(int(val))
        self.apply_pattern(int(val)-1)

    def input_canvas_mouse_down(self, event):
        if self.input_mode != "manual":
            return
        # print('Click at x=', event.x, ', y=', event.y)
        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())
        col = int(event.x // self.neuron_size)
        row = int(event.y // self.neuron_size)

        if row < number_rows and col < number_cols:
            # print("Input layer clicked at", row, col)
            handle = self.input_layer_squares[row][col]
            # print('Create copy of input_layer_excitation')
            self.modified_input_layer_excitation = deepcopy(self.input_layer_excitation)
            self.input_layer_was_copied = True
            if self.input_layer_excitation[row][col] < 0.5:
                self.modified_input_layer_excitation[row][col] = 1.0
                self.canvas_input.itemconfig(handle, fill="black")
            else:
                self.modified_input_layer_excitation[row][col] = 0.0
                self.canvas_input.itemconfig(handle, fill="white")
        else:
            pass
            # print("Missed canvas")

    def input_canvas_drag(self, event):
        if self.input_mode != "manual":
            return
        # print("Mouse dragging over canvas:", event.x, ":", event.y)
        if not self.input_layer_was_copied:
            return
        number_rows = int(g.numRows.get())
        number_cols = int(g.numCols.get())
        col = int(event.x // self.neuron_size)
        row = int(event.y // self.neuron_size)

        if row < number_rows and col < number_cols:
            # print("Input layer dragged at", row, col)
            handle = self.input_layer_squares[row][col]
            self.modified_input_layer_excitation[row][col] = 1.0
            self.canvas_input.itemconfig(handle, fill="black")

    def input_canvas_mouse_up(self, _):
        if self.input_mode != "manual":
            return
        if self.input_layer_was_copied:
            # print('Replace input_layer_excitation')
            self.input_layer_excitation = self.modified_input_layer_excitation
            self.input_layer_was_copied = False
            self.run_network(self.input_layer_excitation.flatten())

    column_1 = 20
    column_2 = column_1 + 130
    column_3 = column_2 + 130
    column_4 = column_3 + 210
    column_5 = column_4 + 90
    column_6 = column_5 + 80

    lineSpace = 40
    header_line = 20
    row_1 = 56
    
    def __init__(self, notebook):
        super(Inspection, self).__init__(notebook, background=g.bgDark)
        self.output_labels = []
        self.rows = [self.row_1 + n * self.lineSpace for n in range(30)]

        # Left Column
        lbl_header_left = tk.Label(self, text='Input Layer', font=g.fontTitle, background=g.bgDark)
        lbl_header_left.place(x=self.column_1, y=self.header_line)

        self.pattern = tk.StringVar()
        lbl_pattern = tk.Label(self, text='Pattern', font=g.fontLabel, background=g.bgDark)
        lbl_pattern.place(x=self.column_1, y=self.rows[5] + 15)
        lbl_value_pattern = tk.Label(self, textvariable=self.pattern,
                                     font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_pattern.place(x=self.column_2, y=self.rows[5] + 15)

        self.canvas_input = tk.Canvas(self, width=200, height=200, bg=g.bgDark, highlightbackground=g.bgDark)
        self.canvas_input.place(x=self.column_1, y=self.rows[0])
        self.canvas_input.bind('<B1-Motion>', self.input_canvas_drag)
        self.canvas_input.bind('<Button-1>', self.input_canvas_mouse_down)
        self.canvas_input.bind('<ButtonRelease-1>', self.input_canvas_mouse_up)
        self.input_layer_squares = []
        self.neuron_size = 0

        # Radio buttons ---------------------------------------------------------------
        self.radio = tk.StringVar()
        self.input_layer_excitation = None
        self.modified_input_layer_excitation = None
        self.input_layer_was_copied = False
        self.rb_manual = tk.Radiobutton(self, text="Manual", variable=self.radio, value="Manual",
                                        command=self.create_manual_pattern, font=g.fontLabel, background=g.bgDark)
        self.rb_manual.place(x=self.column_1, y=self.rows[7])

        self.button_clear = tk.Button(self, text="Clear", width=6, state=tk.DISABLED,
                                      font=g.fontLabel, background=g.bgDark, command=self.clear_manual_pattern)
        self.button_clear.place(x=self.column_2, y=self.rows[7])

        self.input_mode = "record"
        self.rb_record = tk.Radiobutton(self, text="Test Record", variable=self.radio, value="Record",
                                        command=lambda: self.apply_pattern(-1), font=g.fontLabel, background=g.bgDark)
        self.rb_record.place(x=self.column_1, y=self.rows[8])
        self.rb_record.select()

        self.current_record = tk.IntVar()
        self.current_record.set(1)
        self.spin_box_record = tk.Spinbox(self, from_=1, to=1, textvariable=self.current_record,
                                          font=g.fontTitle, width=4, background=g.bgDark, command=self.spin_record)
        self.spin_box_record.place(x=self.column_2, y=self.rows[8] + 3)
        self.spin_box_record.bind('<Return>', self.spin_box_record_changed)

        result_frame = tk.Frame(self, bd=1, bg=g.bgDark, relief=tk.RIDGE)
        result_frame.place(x=self.column_1, y=self.rows[13])
        
        lbl_detected = tk.Label(result_frame, text='Detected:', font=g.fontLabel, background=g.bgDark)
        lbl_detected.grid(row=0, column=0, padx=4, pady=4)

        self.detected = tk.StringVar()
        self.lbl_value_detected = tk.Label(result_frame, textvariable=self.detected,
                                      font=g.fontLabel, width=5, background=g.bgLight)
        self.lbl_value_detected.grid(row=0, column=1, padx=4, pady=4)

        lbl_expected = tk.Label(result_frame, text='Expected:', font=g.fontLabel, background=g.bgDark)
        lbl_expected.grid(row=1, column=0, padx=4, pady=4)

        self.output_index = 0  # The index of the expected winning output neuron
        self.expected = tk.StringVar()
        self.lbl_value_expected = tk.Label(result_frame, textvariable=self.expected,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        self.lbl_value_expected.grid(row=1, column=1, padx=4, pady=4)

        # Middle Column ==============================================================
        self.canvas_hidden = tk.Canvas(self, width=150, height=600, bg=g.bgDark, highlightbackground=g.bgDark)
        self.canvas_hidden.place(x=self.column_3, y=self.rows[0])

        lbl_header_middle = tk.Label(self, text='Hidden Layer', font=g.fontTitle, background=g.bgDark)
        lbl_header_middle.place(x=self.column_3, y=self.header_line)

        # Right Column ==============================================================
        self.canvas_output = tk.Canvas(self, width=150, height=600, bg=g.bgDark, highlightbackground=g.bgDark)
        self.canvas_output.place(x=self.column_5, y=self.rows[0])

        lbl_header_right = tk.Label(self, text='Output Layer', font=g.fontTitle, background=g.bgDark)
        lbl_header_right.place(x=self.column_5, y=self.header_line)



