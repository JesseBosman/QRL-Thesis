from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import random
from fox_in_a_hole_gym import FIAH, Givens
from tools import generate_fiah_transfer_matrix, generate_givens_wall
import numpy as np

class GameGUI:
    def __init__(self):
        self.window = Tk()
        self.window.title("Fox in A Hole")
        self.create_start_menu()
        self.game_state = None
        self.game = None
        self.n_holes = None
        self.game_type = None
    
    def create_start_menu(self):

        self.label_game = Label(self.window, text="Enter the game type (FIAH or Givens):")
        self.label_game.pack()

        self.game_entry = Entry(self.window)
        self.game_entry.pack()

        self.continue_button = Button(self.window, text="Continue", command=self.create_second_menu)
        self.continue_button.pack()

    def create_second_menu(self):
        self.game_type = self.game_entry.get()
        self.destroy_start_menu()
        if self.game_type == "FIAH":
            self.create_fiah_menu()
        elif self.game_type == "Givens":
            self.create_quantum_menu()
        else:
            raise ValueError("Invalid game type")
        
        
        

    def create_fiah_menu(self):
        self.label_holes = Label(self.window, text="Enter the number of holes:")
        self.label_holes.pack()

        self.holes_entry = Entry(self.window)
        self.holes_entry.pack()

        self.start_button = Button(self.window, text="Start", command=self.start_game)
        self.start_button.pack()

    def create_quantum_menu(self):
        self.label_holes = Label(self.window, text="Enter the number of holes:")
        self.label_holes.pack()

        self.holes_entry = Entry(self.window)
        self.holes_entry.pack()

        self.b1_label = Label(self.window, text="Enter the Givens gate for the first layer:")
        self.b1_label.pack()

        self.b1_entry = Entry(self.window)
        self.b1_entry.pack()

        self.t1_label = Label(self.window, text="Enter the rotation in units of Pi for the first layer:")
        self.t1_label.pack()

        self.t1_entry = Entry(self.window)
        self.t1_entry.pack()

        self.b2_label = Label(self.window, text="Enter the Givens gate for the second layer:")
        self.b2_label.pack()

        self.b2_entry = Entry(self.window)
        self.b2_entry.pack()

        self.t2_label = Label(self.window, text="Enter the rotation in units of Pi for the second layer:")
        self.t2_label.pack()
        self.t2_entry = Entry(self.window)
        self.t2_entry.pack()

        self.start_button = Button(self.window, text="Start", command=self.start_game)
        self.start_button.pack()

    def start_game(self):
        if self.game == None:

            n_holes = int(self.holes_entry.get())
            game_type = self.game_type


            if game_type == "FIAH":
                transfer_matrix = generate_fiah_transfer_matrix(n_holes)
                self.game = FIAH(max_steps = 10, transfer_matrices = transfer_matrix)
                self.destroy_fiah_menu()
            elif game_type == "Givens":
                b1 = self.b1_entry.get()
                t1 = float(self.t1_entry.get())*np.pi
                b2 = self.b2_entry.get()
                t2 = float(self.t2_entry.get())*np.pi
                self.transfer_matrix = generate_givens_wall(n_holes, b1, t1, b2, t2)
                self.game = Givens(max_steps = 10, transfer_matrices = self.transfer_matrix)
                self.destroy_quantum_menu()
            else:
                raise ValueError("Invalid game type")

        self.game.reset()
        self.create_game_gui()

    def destroy_fiah_menu(self):
        self.label_holes.destroy()
        self.holes_entry.destroy()
        self.start_button.destroy()
        pass
    def destroy_quantum_menu(self):
        self.label_holes.destroy()
        self.holes_entry.destroy()
        self.b1_label.destroy()
        self.b1_entry.destroy()
        self.t1_label.destroy()
        self.t1_entry.destroy()
        self.b2_label.destroy()
        self.b2_entry.destroy()
        self.t2_label.destroy()
        self.t2_entry.destroy()
        self.start_button.destroy()
        pass
    def destroy_start_menu(self):
        
        self.label_game.destroy()
        self.game_entry.destroy()
        self.continue_button.destroy()

    def destroy_game_gui(self):
        self.intstruction_label.destroy()
        self.guess_counter_label.destroy()
        for button in self.buttons:
            button.destroy()

    def create_game_gui(self):
        self.intstruction_label = Label(self.window, text="Choose a hole to check:")
        self.intstruction_label.pack()

        self.guess_counter = 0
        self.guess_var = IntVar()
        self.guess_var.set(self.guess_counter)
        self.guess_counter_label = Label(self.window, text="Guesses: ", textvariable=self.guess_var)
        self.guess_counter_label.pack()

        self.buttons = []
        for i in range(1, self.game.n_holes + 1):
            button = Button(self.window, text=str(i), command=lambda i=i: self.check_hole(i))
            button.pack()
            self.buttons.append(button)

    def continue_game(self):
        self.destroy_game_gui()
        self.create_game_gui()

    def new_settings(self):
        self.destroy_game_gui()
        self.create_start_menu()
        
    def check_hole(self, hole):

        self.guess_counter += 1
        self.guess_var.set(self.guess_counter)

        self.window.update_idletasks()
        self.game_state, reward, done, _ = self.game.step(hole-1)
        result = "You found the fox!" if reward == 0 else "You did not find the fox."

        messagebox.showinfo("Result", result)
        if reward == 0:
            for button in self.buttons:
                button.config(state=DISABLED)



    def start(self):
        self.window.mainloop()

# Create the GUI for the game
gui = GameGUI()

# Start the game
gui.start()
