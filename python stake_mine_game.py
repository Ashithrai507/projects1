import tkinter as tk
import random

class GuessNumberGame:
    def __init__(self, master):
        self.master = master
        master.title("Guess a Number Game")
        master.geometry("400x350")
        master.configure(bg="#f7f7f7")

        # Initialize game variables
        self.number_to_guess = random.randint(1, 100)
        self.guesses_taken = 0
        self.max_attempts = 10  # Maximum number of guesses allowed

        # Create UI elements
        self.title_label = tk.Label(master, text="Guess a Number!", font=("Helvetica", 20, "bold"), bg="#f7f7f7")
        self.title_label.pack(pady=10)

        self.instruction_label = tk.Label(master, text="Guess a number between 1 and 100", font=("Helvetica", 12), bg="#f7f7f7")
        self.instruction_label.pack(pady=5)

        self.entry = tk.Entry(master, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        self.guess_button = tk.Button(master, text="Guess", command=self.check_guess, bg="#4CAF50", fg="white", font=("Helvetica", 14))
        self.guess_button.pack(pady=5)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 12), bg="#f7f7f7")
        self.result_label.pack(pady=10)

        # Score and Attempts remaining labels
        self.score_label = tk.Label(master, text="Guesses Taken: 0", font=("Helvetica", 12), bg="#f7f7f7")
        self.score_label.pack(pady=5)

        self.attempts_label = tk.Label(master, text=f"Attempts Remaining: {self.max_attempts}", font=("Helvetica", 12), bg="#f7f7f7")
        self.attempts_label.pack(pady=5)

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_game, bg="#f44336", fg="white", font=("Helvetica", 14))
        self.reset_button.pack(pady=5)

        self.feedback_frame = tk.Frame(master, bg="#f7f7f7")
        self.feedback_frame.pack(pady=10)

        self.feedback_label = tk.Label(self.feedback_frame, text="", font=("Helvetica", 12), bg="#f7f7f7")
        self.feedback_label.pack()

        # Bind the Enter key to trigger the check_guess method
        self.master.bind('<Return>', self.check_guess_on_enter)

    def check_guess(self):
        try:
            guess = int(self.entry.get())
            self.guesses_taken += 1
            remaining_attempts = self.max_attempts - self.guesses_taken

            if guess < 1 or guess > 100:
                self.feedback_label.config(text="Please enter a number between 1 and 100!", fg="orange")
            elif guess < self.number_to_guess:
                self.feedback_label.config(text="Too low! Try again.", fg="blue")
            elif guess > self.number_to_guess:
                self.feedback_label.config(text="Too high! Try again.", fg="blue")
            else:
                self.feedback_label.config(text=f"Correct! You found the number in {self.guesses_taken} tries.", fg="green")

            # Update score and attempts labels
            self.score_label.config(text=f"Guesses Taken: {self.guesses_taken}")
            self.attempts_label.config(text=f"Attempts Remaining: {remaining_attempts}")

            # Check if the player has run out of attempts
            if self.guesses_taken >= self.max_attempts:
                self.feedback_label.config(text=f"Game Over! The correct number was {self.number_to_guess}.", fg="red")
                self.guess_button.config(state=tk.DISABLED)  # Disable the guess button if the game is over

        except ValueError:
            self.feedback_label.config(text="Invalid input! Please enter a valid number.", fg="red")

    def check_guess_on_enter(self, event):
        """Allow the Enter key to trigger the guess check."""
        self.check_guess()

    def reset_game(self):
        self.number_to_guess = random.randint(1, 100)
        self.guesses_taken = 0
        self.feedback_label.config(text="")
        self.entry.delete(0, tk.END)
        self.guess_button.config(state=tk.NORMAL)  # Enable the guess button

        # Reset the score and attempts remaining
        self.score_label.config(text="Guesses Taken: 0")
        self.attempts_label.config(text=f"Attempts Remaining: {self.max_attempts}")


# Create the main window
root = tk.Tk()
game = GuessNumberGame(root)

# Start the GUI event loop
root.mainloop()
