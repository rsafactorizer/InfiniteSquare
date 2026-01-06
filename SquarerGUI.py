#!/usr/bin/env python3
"""
Modern Comprehensive GUI for Geometric Lattice Factorization Tool
Dark theme with 160° hue (cyan/teal) accent colors
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import sys
from Squarer import factor_with_lattice_compression

class SquarerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Geometric Lattice Factorization Engine")
        self.root.geometry("1600x1000")
        
        # Color scheme: Dark theme with 160° hue (cyan/teal)
        self.colors = {
            'bg': '#0a0e14',           # Very dark blue-black
            'panel': '#11151c',        # Slightly lighter panel
            'border': '#1a2332',       # Border color
            'text': '#e0f2f1',         # Light text
            'text_dim': '#9ca3af',     # Dimmed text
            'accent': '#00d4aa',       # 160° hue cyan/teal
            'accent_light': '#00ffcc', # Lighter accent
            'accent_dark': '#00a67f',  # Darker accent
            'success': '#00ff88',      # Success green
            'error': '#ff4444',        # Error red
            'warning': '#ffaa00',      # Warning orange
            'info': '#00d4ff',         # Info cyan
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Control variables
        self.is_running = False
        self.output_queue = queue.Queue(maxsize=10000)
        self.worker_thread = None
        
        # Create UI
        self.create_widgets()
        
        # Start output monitor
        self.monitor_output()
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title bar
        title_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(
            title_frame,
            text="🔷 Geometric Lattice Factorization Engine",
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="3D Lattice Compression with Modular Carry & Recursive Refinement",
            font=('Segoe UI', 11),
            bg=self.colors['bg'],
            fg=self.colors['text_dim']
        )
        subtitle.pack(pady=(5, 0))
        
        # Content area: Left panel (controls) + Right panel (output)
        content_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg=self.colors['panel'], relief='flat', bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15), expand=False)
        left_panel.config(width=450)
        
        # Right panel - Output
        right_panel = tk.Frame(content_frame, bg=self.colors['bg'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_control_panel(left_panel)
        self.create_output_panel(right_panel)
    
    def create_control_panel(self, parent):
        """Create the control panel with input fields."""
        # Input section
        input_frame = self.create_section(parent, "Input Parameters", 0)
        
        # N input
        n_label = tk.Label(input_frame, text="Number to Factor (N):", 
                          font=('Segoe UI', 10, 'bold'),
                          bg=self.colors['panel'], fg=self.colors['text'])
        n_label.pack(anchor='w', pady=(0, 5))
        
        self.n_entry = tk.Text(input_frame, height=4, width=50,
                              bg='#0d1117', fg=self.colors['accent'],
                              font=('Consolas', 11),
                              insertbackground=self.colors['accent'],
                              relief='flat', bd=2, highlightthickness=1,
                              highlightbackground=self.colors['border'],
                              highlightcolor=self.colors['accent'])
        self.n_entry.pack(fill=tk.X, pady=(0, 10))
        self.n_entry.insert('1.0', '35')
        
        # Load from file button
        load_btn = tk.Button(input_frame, text="📁 Load from File",
                            command=self.load_from_file,
                            bg=self.colors['border'], fg=self.colors['text'],
                            font=('Segoe UI', 9),
                            relief='flat', cursor='hand2',
                            activebackground=self.colors['accent_dark'],
                            activeforeground='#ffffff',
                            padx=15, pady=5)
        load_btn.pack(anchor='w', pady=(0, 15))
        
        # Lattice size
        lattice_label = tk.Label(input_frame, text="Initial Lattice Size:",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['panel'], fg=self.colors['text'])
        lattice_label.pack(anchor='w', pady=(0, 5))
        
        lattice_frame = tk.Frame(input_frame, bg=self.colors['panel'])
        lattice_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.lattice_size_var = tk.StringVar(value="100")
        lattice_spin = ttk.Spinbox(lattice_frame, from_=10, to=1000,
                                  textvariable=self.lattice_size_var,
                                  width=25, font=('Segoe UI', 10))
        lattice_spin.pack(side=tk.LEFT)
        
        # Iterations
        iter_label = tk.Label(input_frame, text="Recursive Refinement Iterations:",
                            font=('Segoe UI', 10, 'bold'),
                            bg=self.colors['panel'], fg=self.colors['text'])
        iter_label.pack(anchor='w', pady=(15, 5))
        
        self.iterations_var = tk.IntVar(value=10)
        iter_scale = tk.Scale(input_frame, from_=1, to=100,
                             orient=tk.HORIZONTAL,
                             variable=self.iterations_var,
                             bg=self.colors['panel'], fg=self.colors['text'],
                             troughcolor=self.colors['bg'],
                             activebackground=self.colors['accent'],
                             highlightbackground=self.colors['panel'],
                             font=('Segoe UI', 9),
                             length=400)
        iter_scale.pack(fill=tk.X, pady=(0, 5))
        
        iter_value_label = tk.Label(input_frame, 
                                   textvariable=tk.StringVar(value="10"),
                                   font=('Segoe UI', 9),
                                   bg=self.colors['panel'], fg=self.colors['accent'])
        iter_value_label.pack(anchor='w')
        
        # Update label when scale changes
        def update_iter_label(val):
            iter_value_label.config(text=str(int(float(val))))
        iter_scale.config(command=update_iter_label)
        
        # GCD Search Window
        search_label = tk.Label(input_frame, text="GCD Search Window Size:",
                              font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['panel'], fg=self.colors['text'])
        search_label.pack(anchor='w', pady=(15, 5))
        
        search_frame = tk.Frame(input_frame, bg=self.colors['panel'])
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.search_window_var = tk.StringVar(value="10000")
        search_spin = ttk.Spinbox(search_frame, from_=100, to=1000000,
                                 textvariable=self.search_window_var,
                                 width=25, font=('Segoe UI', 10),
                                 increment=1000)
        search_spin.pack(side=tk.LEFT)
        
        search_hint = tk.Label(input_frame, 
                             text="Range: ±N (e.g., 10000 = search ±10000 around target)",
                             font=('Segoe UI', 8),
                             bg=self.colors['panel'], fg=self.colors['text_dim'])
        search_hint.pack(anchor='w', pady=(0, 15))
        
        # Lattice Offset
        offset_label = tk.Label(input_frame, text="Lattice Offset (X, Y, Z):",
                              font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['panel'], fg=self.colors['text'])
        offset_label.pack(anchor='w', pady=(0, 5))
        
        offset_frame = tk.Frame(input_frame, bg=self.colors['panel'])
        offset_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.offset_x_var = tk.StringVar(value="0")
        self.offset_y_var = tk.StringVar(value="0")
        self.offset_z_var = tk.StringVar(value="0")
        
        for i, (label, var) in enumerate([("X:", self.offset_x_var), 
                                          ("Y:", self.offset_y_var), 
                                          ("Z:", self.offset_z_var)]):
            tk.Label(offset_frame, text=label, 
                    font=('Segoe UI', 9),
                    bg=self.colors['panel'], fg=self.colors['text'],
                    width=3).grid(row=0, column=i*2, padx=(0, 5))
            ttk.Spinbox(offset_frame, from_=-10, to=10, textvariable=var,
                       width=8, font=('Segoe UI', 9)).grid(row=0, column=i*2+1, padx=(0, 15))
        
        offset_hint = tk.Label(input_frame, 
                              text="Small offsets break symmetry traps (try ±1 to ±5)",
                              font=('Segoe UI', 8),
                              bg=self.colors['panel'], fg=self.colors['text_dim'])
        offset_hint.pack(anchor='w', pady=(0, 20))
        
        # Control buttons
        button_frame = tk.Frame(input_frame, bg=self.colors['panel'])
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_btn = tk.Button(button_frame, text="▶ Start Factorization",
                                   command=self.start_factorization,
                                   bg=self.colors['accent'], fg='#000000',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat', cursor='hand2',
                                   activebackground=self.colors['accent_light'],
                                   padx=25, pady=15)
        self.start_btn.pack(fill=tk.X, pady=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop",
                                  command=self.stop_factorization,
                                  bg=self.colors['error'], fg='#ffffff',
                                  font=('Segoe UI', 10, 'bold'),
                                  relief='flat', cursor='hand2',
                                  activebackground='#ff6666',
                                  state=tk.DISABLED,
                                  padx=25, pady=12)
        self.stop_btn.pack(fill=tk.X, pady=(0, 10))
        
        clear_btn = tk.Button(button_frame, text="🗑 Clear Output",
                             command=self.clear_output,
                             bg=self.colors['border'], fg=self.colors['text'],
                             font=('Segoe UI', 10),
                             relief='flat', cursor='hand2',
                             activebackground=self.colors['accent_dark'],
                             padx=25, pady=10)
        clear_btn.pack(fill=tk.X)
        
        # Status section
        status_frame = self.create_section(parent, "Status", 15)
        
        self.status_label = tk.Label(status_frame,
                                     text="Ready",
                                     font=('Segoe UI', 11, 'bold'),
                                     bg=self.colors['panel'],
                                     fg=self.colors['accent'])
        self.status_label.pack(anchor='w', pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate',
                                      length=400)
        self.progress.pack(fill=tk.X)
    
    def create_section(self, parent, title, top_pad):
        """Create a section frame with title."""
        frame = tk.Frame(parent, bg=self.colors['panel'], relief='flat', bd=0)
        frame.pack(fill=tk.X, padx=15, pady=(top_pad, 15))
        
        title_label = tk.Label(frame, text=title,
                              font=('Segoe UI', 12, 'bold'),
                              bg=self.colors['panel'],
                              fg=self.colors['accent'])
        title_label.pack(anchor='w', pady=(0, 15))
        
        return frame
    
    def create_output_panel(self, parent):
        """Create the output display panel."""
        # Output header
        header_frame = tk.Frame(parent, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        output_title = tk.Label(header_frame, text="Output & Results",
                               font=('Segoe UI', 14, 'bold'),
                               bg=self.colors['bg'],
                               fg=self.colors['accent'])
        output_title.pack(side=tk.LEFT)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            parent,
            bg='#0d1117',
            fg=self.colors['accent'],
            font=('Consolas', 10),
            insertbackground=self.colors['accent'],
            relief='flat',
            wrap=tk.WORD,
            padx=20,
            pady=20,
            highlightthickness=1,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['accent']
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colored output
        self.output_text.tag_config('success', foreground=self.colors['success'])
        self.output_text.tag_config('error', foreground=self.colors['error'])
        self.output_text.tag_config('warning', foreground=self.colors['warning'])
        self.output_text.tag_config('info', foreground=self.colors['info'])
        self.output_text.tag_config('factor', foreground=self.colors['accent_light'], 
                                   font=('Consolas', 11, 'bold'))
        self.output_text.tag_config('accent', foreground=self.colors['accent'])
    
    def load_from_file(self):
        """Load N from a file."""
        filename = filedialog.askopenfilename(
            title="Select file containing N",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    self.n_entry.delete('1.0', tk.END)
                    self.n_entry.insert('1.0', content)
                    self.log_output(f"✓ Loaded N from {filename}\n", 'success')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def log_output(self, text, tag=''):
        """Add text to output area."""
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_output(self):
        """Clear the output area."""
        self.output_text.delete('1.0', tk.END)
    
    def start_factorization(self):
        """Start the factorization process."""
        if self.is_running:
            return
        
        # Get input
        n_str = self.n_entry.get('1.0', tk.END).strip()
        if not n_str:
            messagebox.showerror("Error", "Please enter a number to factor.")
            return
        
        try:
            N = int(n_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid number format.")
            return
        
        try:
            lattice_size = int(self.lattice_size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid lattice size.")
            return
        
        iterations = self.iterations_var.get()
        
        try:
            search_window = int(self.search_window_var.get())
            if search_window < 100:
                messagebox.showerror("Error", "Search window must be at least 100.")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid search window size.")
            return
        
        try:
            offset_x = int(self.offset_x_var.get())
            offset_y = int(self.offset_y_var.get())
            offset_z = int(self.offset_z_var.get())
            lattice_offset = (offset_x, offset_y, offset_z)
        except ValueError:
            messagebox.showerror("Error", "Invalid lattice offset values.")
            return
        
        # Update UI
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Running...", fg=self.colors['accent'])
        self.progress.start()
        
        # Clear output
        self.clear_output()
        self.log_output("="*80 + "\n", 'accent')
        self.log_output("GEOMETRIC LATTICE FACTORIZATION ENGINE\n", 'accent')
        self.log_output("="*80 + "\n\n", 'accent')
        self.log_output(f"📊 Configuration:\n", 'info')
        self.log_output(f"   Target N: {N}\n", 'accent')
        self.log_output(f"   Bit length: {N.bit_length()} bits\n", 'info')
        self.log_output(f"   Initial lattice size: {lattice_size}×{lattice_size}×{lattice_size}\n", 'info')
        self.log_output(f"   Recursive refinement iterations: {iterations}\n", 'info')
        self.log_output(f"   GCD search window: ±{search_window}\n", 'info')
        self.log_output(f"   Lattice offset: {lattice_offset}\n", 'info')
        self.log_output(f"\n{'='*80}\n", 'accent')
        self.log_output("🚀 Starting factorization process...\n\n", 'info')
        
        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self.factorization_worker,
            args=(N, lattice_size, iterations, search_window, lattice_offset),
            daemon=True
        )
        self.worker_thread.start()
    
    def stop_factorization(self):
        """Stop the factorization process."""
        self.is_running = False
        self.status_label.config(text="Stopping...", fg=self.colors['warning'])
    
    def factorization_worker(self, N, lattice_size, iterations, search_window, lattice_offset):
        """Worker thread for factorization with verbose real-time output."""
        try:
            # Create a custom stdout that writes to queue immediately
            class QueueWriter:
                def __init__(self, queue):
                    self.queue = queue
                
                def write(self, text):
                    if text:
                        try:
                            self.queue.put(('output', text), block=False)
                        except queue.Full:
                            pass
                
                def flush(self):
                    pass
            
            # Send initial messages BEFORE redirecting stdout
            try:
                self.output_queue.put(('status', 'Initializing factorization...'), block=False)
                self.output_queue.put(('output', f'[CONFIG] Using {iterations} recursive refinement iterations\n'), block=False)
                self.output_queue.put(('output', f'[CONFIG] GCD search window: ±{search_window}\n'), block=False)
                self.output_queue.put(('output', f'[CONFIG] Lattice offset: {lattice_offset}\n'), block=False)
            except queue.Full:
                pass
            
            # Redirect stdout to queue writer
            old_stdout = sys.stdout
            queue_writer = QueueWriter(self.output_queue)
            sys.stdout = queue_writer
            
            # Force immediate flush
            sys.stdout.flush()
            
            # Send status update
            try:
                self.output_queue.put(('status', 'Running factorization...'), block=False)
            except queue.Full:
                pass
            
            # Call factorization
            result = factor_with_lattice_compression(
                N, 
                lattice_size=lattice_size, 
                zoom_iterations=iterations, 
                search_window_size=search_window, 
                lattice_offset=lattice_offset
            )
            
            # Flush any remaining output
            queue_writer.flush()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Send completion status
            self.output_queue.put(('status', 'Factorization complete'))
            self.output_queue.put(('result', result))
            self.output_queue.put(('done', None))
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.output_queue.put(('error', error_msg))
            self.output_queue.put(('status', 'Error occurred'))
            self.output_queue.put(('done', None))
    
    def monitor_output(self):
        """Monitor output queue and update UI with verbose real-time updates."""
        try:
            # Process multiple messages per cycle
            for _ in range(100):
                try:
                    msg_type, data = self.output_queue.get_nowait()
                except queue.Empty:
                    break
                
                if msg_type == 'output':
                    # Auto-detect message types for color coding
                    data_lower = data.lower()
                    if '✓' in data or 'factors found' in data_lower or 'found factor' in data_lower:
                        self.log_output(data, 'success')
                    elif 'error' in data_lower or 'traceback' in data_lower or 'exception' in data_lower:
                        self.log_output(data, 'error')
                    elif 'warning' in data_lower:
                        self.log_output(data, 'warning')
                    elif any(kw in data_lower for kw in ['stage', 'iteration', 'performing', 'compressed', 'handoff', 'macro', 'micro', 'collapse']):
                        self.log_output(data, 'info')
                    elif '=' in data and len(data.strip()) > 10:
                        self.log_output(data, 'accent')
                    elif any(kw in data_lower for kw in ['modular', 'remainder', 'gcd', 'search', 'resonance', 'offset', 'precision', 'encoding', 'lattice', 'zoom', 'refinement']):
                        self.log_output(data, 'info')
                    elif any(kw in data_lower for kw in ['config', 'using', 'window', 'coordinate', 'shadow', 'carry']):
                        self.log_output(data, 'info')
                    else:
                        self.log_output(data)
                elif msg_type == 'status':
                    self.status_label.config(text=data, fg=self.colors['accent'])
                    self.log_output(f"[STATUS] {data}\n", 'info')
                elif msg_type == 'result':
                    self.display_results(data)
                elif msg_type == 'error':
                    self.log_output(f"\n{'='*80}\n", 'error')
                    self.log_output(f"ERROR: {data}\n", 'error')
                    self.log_output(f"{'='*80}\n", 'error')
                elif msg_type == 'done':
                    self.factorization_done()
                    
        except queue.Empty:
            pass
        except Exception:
            pass
        
        # Schedule next check (10ms for real-time feel)
        self.root.after(10, self.monitor_output)
    
    def display_results(self, result):
        """Display factorization results."""
        self.log_output("\n" + "="*80 + "\n", 'accent')
        self.log_output("📈 FACTORIZATION RESULTS\n", 'accent')
        self.log_output("="*80 + "\n\n", 'accent')
        
        if result and 'factors' in result:
            factors = result['factors']
            if factors:
                self.log_output("✅ FACTORS FOUND:\n\n", 'success')
                for i, factor_pair in enumerate(factors, 1):
                    p, q = factor_pair
                    verified = p * q == result.get('N', 0)
                    self.log_output(f"  Factor Pair #{i}:\n", 'info')
                    self.log_output(f"    p = {p}\n", 'factor')
                    self.log_output(f"    q = {q}\n", 'factor')
                    self.log_output(f"    p × q = {p * q}\n", 'factor')
                    self.log_output(f"    Verification: {'✓ CORRECT' if verified else '✗ INCORRECT'}\n\n", 
                                  'success' if verified else 'error')
            else:
                self.log_output("❌ No factors found.\n", 'error')
        else:
            self.log_output("❌ Factorization failed.\n", 'error')
    
    def factorization_done(self):
        """Handle factorization completion."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()
        self.status_label.config(text="Ready", fg=self.colors['accent'])


if __name__ == "__main__":
    root = tk.Tk()
    app = SquarerGUI(root)
    root.mainloop()
