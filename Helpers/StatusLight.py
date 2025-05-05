import tkinter as tk
from tkinter import ttk

class StatusLight(ttk.Frame):
    def __init__(self, parent, size=15, **kwargs):
        ttk.Frame.__init__(self, parent, width=size, height=size, **kwargs)
        self.size = size

        self.canvas = tk.Canvas(self, width=size, height=size,
                                highlightthickness=0, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.state = "off"
        self.update_color()

    def set_state(self, state):
        """Set light state: 'off', 'loading', 'ready', 'active'"""
        self.state = state
        self.update_color()

    def update_color(self):
        color_map = {
            "off": "#666666",
            "loading": "#ff3333",
            "ready": "#ffcc00",
            "active": "#33cc33"
        }
        color = color_map.get(self.state, "#666666")

        self.canvas.delete("all")
        self.canvas.create_oval(2, 2, self.size - 2, self.size - 2,
                                fill=color, outline="#444444", width=1)