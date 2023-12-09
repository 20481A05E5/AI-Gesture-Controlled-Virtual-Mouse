
from tkinter import *
from chat import get_response, bot_name
from tkinter import scrolledtext
import speech_recognition as sr
import threading
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        self.recognizer = sr.Recognizer()  # Initialize recognizer here
        self.microphone_available = True  # Flag to check if a microphone is available
        try:
            sr.Microphone()
            
        except OSError:
            self.microphone_available = False

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = scrolledtext.ScrolledText(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                                     font=FONT, padx=5, pady=5, wrap=WORD)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # bottom label (now an instance variable)
        self.bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        self.bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(self.bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()

        # send button
        send_button = Button(self.bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

        # record button
        record_button = Button(self.bottom_label, text="Speak", font=FONT_BOLD, width=20, bg=BG_GRAY,
                                command=self._start_recording)
        record_button.place(relx=0.56, rely=0.008, relheight=0.06, relwidth=0.2)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        out=get_response(msg)
        if out!=None:
            msg2 = f"{bot_name}: {out}\n\n"
            self.text_widget.configure(state=NORMAL)
            self.text_widget.insert(END, msg2)
            self.text_widget.configure(state=DISABLED)
            
            self.text_widget.see(END)
             
    def _start_recording(self):
        if not self.microphone_available:
            self._insert_message("No microphone available.", bot_name)
            return

        # Start the recording in a separate thread
        threading.Thread(target=self.recognize_speech).start()

    def recognize_speech(self):
        # Create a speech recognition object
        recognizer = sr.Recognizer()

        # Capture audio from the microphone
        with sr.Microphone() as source:
            self._insert_message("Listening...", bot_name)
            audio = recognizer.listen(source, timeout=5)

        try:
            # Use Google Web Speech API to recognize the speech
            text = recognizer.recognize_google(audio)
            self._insert_message(text, "You")
        except sr.UnknownValueError:
            self._insert_message("Unable to understand what you just said.", bot_name)
        except sr.RequestError as e:
            print(f"Error making the request: {e}")
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
