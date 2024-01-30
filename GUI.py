import tkinter
import tkinter.messagebox
import customtkinter
from customtkinter import *
from fox_in_a_hole_gym import FIAH, Givens
from tools import generate_fiah_transfer_matrix, generate_givens_wall
import numpy as np

class GameGUI(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Fox in A Hole")
        self.create_start_menu()
        self.game_state = None
        self.game = None
        self.n_holes = None
        self.game_type = None
    
    def create_start_menu(self):

        self.label_game = CTkLabel(self, text="Select the game type:")
        self.label_game.grid(row=0, column=0, columnspan=2)

        self.fiah_button = CTkButton(self, text="Classical", command=self.create_fiah_menu)
        self.fiah_button.grid(row=1, column=0)

        self.givens_button = CTkButton(self, text="Quantum", command=self.create_quantum_menu)
        self.givens_button.grid(row=1, column=1)    
    
    def destroy_start_menu(self):
        
        self.label_game.destroy()
        self.fiah_button.destroy()
        self.givens_button.destroy()
        

    def create_fiah_menu(self):
        self.destroy_start_menu()
        self.game_type = "FIAH"
        self.label_holes = CTkLabel(self, text="Enter the number of holes:")
        self.label_holes.pack()

        self.holes_entry = CTkEntry(self)
        self.holes_entry.pack()

        self.start_button = CTkButton(self, text="Start", command=self.start_game)
        self.start_button.pack()

    def create_quantum_menu(self):
        self.destroy_start_menu()
        self.game_type = "Givens"
        self.label_holes = CTkLabel(self, text="Enter the number of holes:")
        self.label_holes.pack()

        self.holes_entry = CTkEntry(self)
        self.holes_entry.pack()

        self.b1_label = CTkLabel(self, text="Enter the Givens gate for the first layer:")
        self.b1_label.pack()

        self.b1_entry = CTkEntry(self)
        self.b1_entry.pack()

        self.t1_label = CTkLabel(self, text="Enter the rotation in units of Pi for the first layer:")
        self.t1_label.pack()

        self.t1_entry = CTkEntry(self)
        self.t1_entry.pack()

        self.b2_label = CTkLabel(self, text="Enter the Givens gate for the second layer:")
        self.b2_label.pack()

        self.b2_entry = CTkEntry(self)
        self.b2_entry.pack()

        self.t2_label = CTkLabel(self, text="Enter the rotation in units of Pi for the second layer:")
        self.t2_label.pack()
        self.t2_entry = CTkEntry(self)
        self.t2_entry.pack()

        self.start_button = CTkButton(self, text="Start", command=self.start_game)
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

    def destroy_game_gui(self):
        self.guesses_label.destroy()
        self.intstruction_label.destroy()
        self.guess_counter_label.destroy()
        for hole in self.holes:
            hole.destroy()

    def create_game_gui(self):
        self.guesses_label = CTkLabel(self, text="Number of guesses:")
        self.guesses_label.grid(row=0, column=0, columnspan=self.game.n_holes)
        self.guess_counter = 0
        self.guess_var = IntVar()
        self.guess_var.set(self.guess_counter)
        self.guess_counter_label = CTkLabel(self, textvariable=self.guess_var)
        self.guess_counter_label.grid(row=1, column=0, columnspan=self.game.n_holes)
        
        self.intstruction_label = CTkLabel(self, text="Choose a hole to check:")
        self.intstruction_label.grid(row=2, column=0, columnspan=self.game.n_holes)
        self.holes = []
        for i in range(1, self.game.n_holes + 1):
            hole = CTkButton(self, text=str(i), command=lambda i=i: self.check_hole(i))
            hole.grid(row=3, column=i-1)
            self.holes.append(hole)

    def new_settings(self):
        self.decision_label.destroy()
        self.exit_button.destroy()
        self.new_button.destroy()
        self.continue_button.destroy()
        self.create_start_menu()
    
    def new_game(self):
        self.decision_label.destroy()
        self.exit_button.destroy()
        self.new_button.destroy()
        self.continue_button.destroy()
        self.create_game_gui()
        
    def check_hole(self, guess):
        self.guess_counter += 1
        self.guess_var.set(self.guess_counter)

        self.update_idletasks()
        self.game_state, reward, done, _ = self.game.step(guess-1)

        if reward == 0:
            for hole in self.holes:
                hole.configure(state=DISABLED)
            tkinter.messagebox.showinfo("Result", "You found the fox in " + str(self.guess_counter) + " guesses!")
            self.destroy_game_gui()
            self.decision_label = CTkLabel(self, text="Do you want to play again?")
            self.decision_label.grid(row=0, column=0, columnspan=3)
            self.exit_button = CTkButton(self, text="Quit", command=self.destroy)
            self.exit_button.grid(row=1, column=0)
            self.new_button = CTkButton(self, text="New Settings", command=self.new_settings)
            self.new_button.grid(row=1, column=1)
            self.continue_button = CTkButton(self, text="Continue", command=self.new_game)
            self.continue_button.grid(row=1, column=2)

        else:
            tkinter.messagebox.showinfo("Result", "You did not find the fox. Try again.")




    def start(self):
        self.mainloop()

# Create the GUI for the game
gui = GameGUI()

# Start the game
gui.start()
