import tkinter as tk
import numpy as np
from tkinter import messagebox
import Global as g

def load_data_file(file_path, target):
    file = open(file_path)
    try:
        header = file.readline(200)
        entries = header.split(" ", 10)
        attributes = dict()
        for e in entries:
            elements = e.split("=")
            attributes[elements[0]] = elements[1].rstrip()

        if attributes["application"] != "NetLab":
            tk.messagebox.showinfo('Incompatible file content', header)
            return False, 0

        if attributes["type"] != "data_bw" and attributes["type"] != "data_gray":
            diagnosis = "Expected type=data, received: " + attributes["type"]
            tk.messagebox.showinfo('Incompatible file content', diagnosis)
            return False, 0

        num_rows = int(attributes["input_rows"])
        g.numRows.set(num_rows)
        num_cols = int(attributes["input_columns"])
        g.numCols.set(num_cols)
        file_format = attributes["version"]

        if file_format != '0':
            tk.messagebox.showinfo('Cannot read file format', file_format)
            return False, 0

    except:
        tk.messagebox.showinfo('Unexpected header line', header)
        return False, 0

    try:
        second_line = file.readline(200)
        entries = second_line.rstrip().split(" ", 200)

        if entries[0] != "patterns:":
            tk.messagebox.showinfo('Incompatible file content, second line is:', second_line)
            return False, 0

        output_layer_size = len(entries) - 1
        g.outputLayerSize.set(output_layer_size)
        g.outputLayerLabels = entries[1:]

    except:
        tk.messagebox.showinfo('Unexpected second line', second_line)
        return False, 0

    number_records = read_format_string(file, attributes["type"], num_rows, num_cols, target)

    if number_records is None:
        return False, 0
    else:
        return True, number_records


def read_format_string(file, type, rows, columns, target):
    if type == "data_bw":
        t = 0
    elif type == "data_gray":
        t = 1
    else:
        tk.messagebox.showerror('Unexpected data type', type)
        return

    target.clear()  # Clear old pattern data
    block_len = rows + 1
    line_no = 0
    number_records = 0
    pattern_line = []
    for line in file:
        offset = line_no % block_len
        if offset == 0:
            label = line.rstrip()
            try:
                label_index = g.outputLayerLabels.index(label)
            except ValueError:
                msg = 'Unexpected label in line ' + str(line_no + 3) + ": '" + label + "'"
                tk.messagebox.showerror('Error in training data', msg)
                return
            line_no += 1
            number_records += 1
            continue
        else:
            arr = [None] * columns
            if t == 0:  # black and white
                for col in range(columns):
                    mark = line[col]
                    if mark == "x":
                        arr[col] = 1.0
                    else:
                        arr[col] = 0.0
            elif t == 1:  # gray scale
                pixels = line.split()
                pixels[len(pixels) - 1].rstrip()
                for col in range(columns):
                    arr[col] = float(pixels[col]) / 255.0

        pattern_line.append(arr)
        if offset == rows:
            pattern = np.asarray(pattern_line).ravel()
            target.append((label_index, pattern))
            pattern_line = []
        line_no += 1
    return number_records

