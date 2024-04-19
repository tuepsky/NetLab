import json
import tkinter as tk
import tkinter.scrolledtext as tkscrolled
import numpy as np
import scipy.special
import time
import matplotlib.pyplot as plt
import Global as g
from NeuroNet import NeuroNet
from Confusion import ConfusionMatrix
import FileAccess

class SetTrainTest(tk.Frame):

    def run_training(self):
        g.gui.update_status('Training in progress, please wait!')
        # Check parameter
        try:
            alpha = float(self.alpha.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Alpha has an invalid value')
            return
        try:
            epochs = int(self.epochs.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Epochs has an invalid value')
            return
        try:
            hidden_layer_size = int(self.hiddenLayerSize.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Hidden layer size has an invalid value')
            return
        try:
            output_layer_size = int(g.outputLayerSize.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Output layer size has an invalid value')
            return
        try:
            random_seed = int(self.randomSeed.get())
        except ValueError:
            tk.messagebox.showinfo('Setup error:', 'Random seed has an invalid value')
            return

        if not g.neuroNet:
            num_rows = int(g.numRows.get())
            num_cols = int(g.numCols.get())
            g.neuroNet = NeuroNet(
                num_rows * num_cols,  # = input_layer_size \
                hidden_layer_size,
                output_layer_size,
                random_seed)

        all_pattern = g.all_training_patterns
        self.all_errors = []
        start_time = time.perf_counter()
        for e in range(epochs):
            self.current_epoch.set(str(e+1))
            self.update()
            errors = g.neuroNet.train(all_pattern, alpha)
            self.all_errors.append(errors)
            elapsed = time.perf_counter() - start_time
            self.training_time.set(int(elapsed))
            self.update()
        if self.all_errors:
            self.last_error.set("%6.4f" % self.all_errors[-1])
        g.gui.update_status('')

    def run_test(self):
        if len(g.all_test_patterns) > 0:
            all_pattern = g.all_test_patterns
        else:
            tk.messagebox.showinfo('No test patterns loaded', 'Using training patterns')
            all_pattern = g.all_training_patterns

        g.gui.update_status('Test in progress, please wait!')

        pattern_index = 1
        failing_pattern_indexes = []
        n_labels = len(g.training_pattern_labels)
        self.confusion = [[0 for c in range(n_labels)]
                          for r in range(n_labels)]
        for pattern in all_pattern:
            output_index, data = pattern
            output_index = int(output_index)
            hidden_excitation, output_excitation = g.neuroNet.run(data)
            max_excitation = -100
            for i, ex in enumerate(output_excitation):
                if ex > max_excitation:
                    max_excitation = ex
                    index_detected_label = i
            self.confusion[output_index][index_detected_label] += 1        
            if index_detected_label != output_index:
                failing_pattern_indexes.append(str(pattern_index))
            pattern_index += 1
        failure_rate = len(failing_pattern_indexes) / len(all_pattern) * 100
        self.failing_tests.set(str(len(failing_pattern_indexes)))
        self.performance.set("%4.2f" % (100 - failure_rate) + "%")
        self.failure_rate.set("%4.2f" % failure_rate + "%")
        failing_pattern = ", ".join(failing_pattern_indexes)
        self.text_value_failing_records.config(state=tk.NORMAL)
        self.text_value_failing_records.delete(1.0, tk.END)
        self.text_value_failing_records.insert(tk.END, failing_pattern)
        self.text_value_failing_records.config(state=tk.DISABLED)
        g.gui.update_status('')

    def show_error_curve(self):
        X = np.linspace(1, len(self.all_errors), num=len(self.all_errors))
        plt.plot(X, self.all_errors)
        plt.show()

    def reset(self):
        g.neuroNet = None
        self.all_errors = None
        self.current_epoch.set("")
        self.last_error.set("")
        self.training_time.set("")

    def load_train_data(self):
        files = tk.filedialog.askopenfilenames()
        if len(files) == 0:  # User cancelled
            return

        g.neuroNet = None
        
        # Clear test patterns
        g.all_test_patterns = []
        g.numberTestRecords.set('')
        self.failing_tests.set('')
        self.performance.set('')
        self.failure_rate.set('')
        
        g.training_file_name.set("Training data: " + files[0])
        g.gui.update_status('Loading training data, please wait!')
        success, number_records = \
                 FileAccess.load_data_file(files[0], g.all_training_patterns,
                                           g.training_pattern_labels)
        
        output_layer_size = len(g.training_pattern_labels)
        g.outputLayerSize.set(output_layer_size)
        g.gui.update_status('')
        g.numberTrainingRecords.set(str(number_records))
                
    def load_test_data(self):
        files = tk.filedialog.askopenfilenames()
        if len(files) == 0:  # User cancelled
            return

        if len(g.all_training_patterns) == 0:
            tk.messagebox.showinfo('Setup error:', 'Please load training pattern first')
            return

        g.test_file_name.set("Test data: " + files[0])
        g.gui.update_status('Loading test data, please wait!')

        success, number_records = \
                 FileAccess.load_data_file(files[0], g.all_test_patterns,
                                           g.test_pattern_labels)
        g.gui.update_status('')
        g.numberTestRecords.set(str(number_records))

    def save_network(self):
        if g.neuroNet:
            data = g.neuroNet.dump()
            outfile = open('network.txt', 'w')
            json.dump(data, outfile, indent=2)
            outfile.close()
        else:
            tk.messagebox.showinfo('Warning:', 'No network available')      

    def show_confusion(self):
        if self.confusion:
            ConfusionMatrix(g.gui, g.training_pattern_labels, self.confusion)
    
    # GUI
    column_1 = 20
    column_2 = column_1 + 150
    column_3 = column_2 + 110
    column_4 = column_3 + 150
    column_5 = column_4 + 110
    column_6 = column_5 + 140
    column_7 = column_6 + 60

    lineSpace = 40
    header_line = 20
    row_1 = 60
    
    def __init__(self, notebook):
        g.numRows = tk.StringVar()
        g.numCols = tk.StringVar()
        g.outputLayerSize = tk.StringVar()
        g.numberTestRecords = tk.StringVar()
        g.numberTrainingRecords = tk.StringVar()
        self.alpha = tk.StringVar()
        self.alpha.set("0.2")
        self.epochs = tk.StringVar()
        self.epochs.set("50")
        self.randomSeed = tk.StringVar()
        self.randomSeed.set("1")
        self.hiddenLayerSize = tk.StringVar()
        self.hiddenLayerSize.set("20")
        self.neuron_net_initialized = False
        self.all_errors = None
        self.confusion = None
        self.current_epoch = tk.StringVar()
        self.last_error = tk.StringVar()
        self.failing_tests = tk.StringVar()
        self.performance = tk.StringVar()
        self.failure_rate = tk.StringVar()
        self.training_time = tk.StringVar()

        super(SetTrainTest, self).__init__(notebook, background=g.bgDark)
        self.rows = [self.row_1 + n * self.lineSpace for n in range(13)]

        # Left Column
        lbl_header_left = tk.Label(self, text='Setup', font=g.fontTitle, background=g.bgDark)
        lbl_header_left.place(x=self.column_1, y=self.header_line)

        # Load trainig data
        button_Load_train = tk.Button(self, text="Load Training Data", width=21,
                               font=g.fontLabel, background=g.bgDark, command=self.load_train_data)
        button_Load_train.place(x=self.column_1, y=self.rows[0] - 5)

        # Input Layer Width
        lbl_input_layer_width = tk.Label(self, text='Input Layer Width', font=g.fontLabel, background=g.bgDark)
        lbl_input_layer_width.place(x=self.column_1, y=self.rows[1])
        lbl_value_input_layer_width = tk.Label(self, textvariable=g.numCols,
                                               font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_input_layer_width.place(x=self.column_2, y=self.rows[1])

        # Input Layer Height
        lbl_input_layer_height = tk.Label(self, text='Input Layer Height', font=g.fontLabel, background=g.bgDark)
        lbl_input_layer_height.place(x=self.column_1, y=self.rows[2])
        lbl_value_input_layer_height = tk.Label(self, textvariable=g.numRows,
                                                font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_input_layer_height.place(x=self.column_2, y=self.rows[2])

        # Output layer size
        lbl_output_layer_size = tk.Label(self, text='Output Layer Size', font=g.fontLabel, background=g.bgDark)
        lbl_output_layer_size.place(x=self.column_1, y=self.rows[3])
        val_output_layer_size = tk.Label(self, textvariable=g.outputLayerSize, justify=tk.CENTER,
                                         font=g.fontLabel, width=5, background=g.bgLight)
        val_output_layer_size.place(x=self.column_2, y=self.rows[3])

        # Hidden layer size
        lbl_hidden_layer_size = tk.Label(self, text='Hidden Layer Size', font=g.fontLabel, background=g.bgDark)
        lbl_hidden_layer_size.place(x=self.column_1, y=self.rows[4])
        e_hidden_layer_size = tk.Entry(self, textvariable=self.hiddenLayerSize, justify=tk.CENTER,
                                       font=g.fontLabel, width=5, background=g.bgBlue)
        e_hidden_layer_size.place(x=self.column_2, y=self.rows[4])

        # Alpha
        lbl_step_width = tk.Label(self, text='Step Width (alpha)', font=g.fontLabel, background=g.bgDark)
        lbl_step_width.place(x=self.column_1, y=self.rows[5])
        e_alpha = tk.Entry(self, textvariable=self.alpha, justify=tk.CENTER,
                           font=g.fontLabel, width=5, background=g.bgBlue)
        e_alpha.place(x=self.column_2, y=self.rows[5])

        # Epochs
        lbl_epochs = tk.Label(self, text='Epochs', font=g.fontLabel, background=g.bgDark)
        lbl_epochs.place(x=self.column_1, y=self.rows[6])
        e_epochs = tk.Entry(self, textvariable=self.epochs, justify=tk.CENTER,
                            font=g.fontLabel, width=5, background=g.bgBlue)
        e_epochs.place(x=self.column_2, y=self.rows[6])

        # Random Seed
        lbl_random = tk.Label(self, text='Random Seed', font=g.fontLabel, background=g.bgDark)
        lbl_random.place(x=self.column_1, y=self.rows[7])
        e_random = tk.Entry(self, textvariable=self.randomSeed, justify=tk.CENTER,
                            font=g.fontLabel, width=5, background=g.bgBlue)
        e_random.place(x=self.column_2, y=self.rows[7])

        # Middle Column ==============================================================
        lbl_header_middle = tk.Label(self, text='Train', font=g.fontTitle, background=g.bgDark)
        lbl_header_middle.place(x=self.column_3, y=self.header_line)

        # Run training button
        button_run = tk.Button(self, text="Run Training", width=21,
                               font=g.fontLabel, background=g.bgDark, command=self.run_training)
        button_run.place(x=self.column_3, y=self.rows[0] - 5)

        # Number of training records
        lbl_train_recs = tk.Label(self, text='Training Records', font=g.fontLabel, background=g.bgDark)
        lbl_train_recs.place(x=self.column_3, y=self.rows[1])
        lbl_value_train_recs = tk.Label(self, textvariable=g.numberTrainingRecords,
                                        font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_train_recs.place(x=self.column_4, y=self.rows[1])

        # Current Epoch
        lbl_current_epoch = tk.Label(self, text='Current Epoch', font=g.fontLabel, background=g.bgDark)
        lbl_current_epoch.place(x=self.column_3, y=self.rows[2])
        lbl_value_current_epoch = tk.Label(self, textvariable=self.current_epoch,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_current_epoch.place(x=self.column_4, y=self.rows[2])

        # Last error
        lbl_last_error = tk.Label(self, text='Last Error', font=g.fontLabel, background=g.bgDark)
        lbl_last_error.place(x=self.column_3, y=self.rows[3])
        lbl_value_last_error = tk.Label(self, textvariable=self.last_error,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_last_error.place(x=self.column_4, y=self.rows[3])

        # Training duration
        lbl_training_duration = tk.Label(self, text='Time Spent [sec]', font=g.fontLabel, background=g.bgDark)
        lbl_training_duration.place(x=self.column_3, y=self.rows[4])
        lbl_value_training_duration = tk.Label(self, textvariable=self.training_time,
                                           font=g.fontLabel, width=5, background=g.bgLight)
        lbl_value_training_duration.place(x=self.column_4, y=self.rows[4])

        # Show Error Curve
        button_error_curve = tk.Button(self, text="Show Error Curve", width=21, font=g.fontLabel, background=g.bgDark,
                                       command=self.show_error_curve)
        button_error_curve.place(x=self.column_3, y=self.rows[5])

        # Reset button
        button_reset = tk.Button(self, text="Reset Neural Network", width=21,
                                 font=g.fontLabel, background=g.bgDark, command=self.reset)
        button_reset.place(x=self.column_3, y=self.rows[6])

        # Store button
        button_store = tk.Button(self, text="Save Network", width=21,
                                 font=g.fontLabel, background=g.bgDark, command=self.save_network)
        button_store.place(x=self.column_3, y=self.rows[7])
       
        # Right Column ==============================================================
        lbl_header_right = tk.Label(self, text='Test', font=g.fontTitle, background=g.bgDark)
        lbl_header_right.place(x=self.column_5, y=self.header_line)

        # Load test data
        button_Load_test = tk.Button(self, text="Load Test Data", width=21,
                               font=g.fontLabel, background=g.bgDark, command=self.load_test_data)
        button_Load_test.place(x=self.column_5, y=self.rows[0] - 5)
        
        # Run test button
        button_run = tk.Button(self, text="Run Test", width=21,
                               font=g.fontLabel, background=g.bgDark, command=self.run_test)
        button_run.place(x=self.column_5, y=self.rows[1] - 5)

        # Number of test records
        lbl_failure_rate = tk.Label(self, text='Test Records', font=g.fontLabel, background=g.bgDark)
        lbl_failure_rate.place(x=self.column_5, y=self.rows[2])
        lbl_value_failure_rate = tk.Label(self, textvariable=g.numberTestRecords,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_failure_rate.place(x=self.column_6, y=self.rows[2])

        # Number of failing tests
        lbl_failing_tests = tk.Label(self, text='Failing Tests', font=g.fontLabel, background=g.bgDark)
        lbl_failing_tests.place(x=self.column_5, y=self.rows[3])
        lbl_value_failing_tests = tk.Label(self, textvariable=self.failing_tests,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_failing_tests.place(x=self.column_6, y=self.rows[3])
        
        # Performance
        lbl_performance = tk.Label(self, text='Performance', font=g.fontLabel, background=g.bgDark)
        lbl_performance.place(x=self.column_5, y=self.rows[4])
        lbl_value_performance = tk.Label(self, textvariable=self.performance,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_performance.place(x=self.column_6, y=self.rows[4])

        # Failure Rate
        lbl_failure_rate = tk.Label(self, text='Failure Rate', font=g.fontLabel, background=g.bgDark)
        lbl_failure_rate.place(x=self.column_5, y=self.rows[5])
        lbl_value_failure_rate = tk.Label(self, textvariable=self.failure_rate,
                                          font=g.fontLabel, width=6, background=g.bgLight)
        lbl_value_failure_rate.place(x=self.column_6, y=self.rows[5])

        # Failing Records
        lbl_failing_records = tk.Label(self, text='Failing Records:', font=g.fontLabel, background=g.bgDark)
        lbl_failing_records.place(x=self.column_5, y=self.rows[6])

        self.text_value_failing_records = \
            tkscrolled.ScrolledText(self, font=g.fontLabel, height=10, width=20, background=g.bgLight,
                                    relief=tk.FLAT, state=tk.DISABLED,  wrap=tk.WORD)
        self.text_value_failing_records.place(x=self.column_5, y=self.rows[7])

        # Show confusion matrix
        button_show = tk.Button(self, text="Show Confusion Matrix", width=21,
                                font=g.fontLabel, background=g.bgDark, command=self.show_confusion)
        button_show.place(x=self.column_5, y=self.rows[12])