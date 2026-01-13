#!/usr/bin/env python3
"""
Spin chain: real-time trajectory (master-equation dynamics)
- Single stochastic trajectory (Gillespie-like) from initial bitstring
- Controls: gamma_plus, gamma_minus, L, Speed
- Checkbox to toggle Periodic Boundary Conditions (PBC) vs Open boundaries
- Fixed UI layout so the Speed controller doesn't overlap text

Run: python spin_chain_pbc_speed_fixed.py
Requires: pygame, numpy (numpy used lightly; not required for core logic)
"""
import sys
import time
import math
import random
import numpy as np
import pygame

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_L = 80
MIN_L = 3
MAX_L = 600
WINDOW_W = 1200
WINDOW_H = 720
FPS = 60

# Colors
WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
BLUE = (60, 140, 240)   # spin 0
RED = (220, 60, 60)     # spin 1
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
SLIDER_BG = (235, 235, 235)
BUTTON_BG = (200, 220, 200)
CHECK_BG = (230, 230, 230)
CHECK_BORDER = (80, 80, 80)

# ---------------------------
# UI widgets
# ---------------------------
class Button:
    def __init__(self, rect, text, font, bg=BUTTON_BG):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.bg = bg

    def draw(self, surf):
        pygame.draw.rect(surf, self.bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, DARK_GRAY, self.rect, 2, border_radius=6)
        txt = self.font.render(self.text, True, BLACK)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    def __init__(self, rect, min_val, max_val, value, font, label=''):
        self.rect = pygame.Rect(rect)
        self.min = min_val
        self.max = max_val
        self.value = float(value)
        self.font = font
        self.label = label
        self.dragging = False
        self.knob_radius = 7

    def draw(self, surf):
        pygame.draw.rect(surf, SLIDER_BG, self.rect, border_radius=6)
        pygame.draw.rect(surf, DARK_GRAY, self.rect, 1, border_radius=6)
        t = (self.value - self.min) / (self.max - self.min) if self.max > self.min else 0.0
        x = self.rect.left + int(t * (self.rect.width - 2*self.knob_radius)) + self.knob_radius
        y = self.rect.centery
        pygame.draw.circle(surf, DARK_GRAY, (x, y), self.knob_radius+2)
        pygame.draw.circle(surf, WHITE, (x, y), self.knob_radius)
        # label above slider
        label_s = self.font.render(f"{self.label}: {self.value:.3g}", True, BLACK)
        surf.blit(label_s, (self.rect.left, self.rect.top - 22))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.dragging = True
            self.set_from_mouse(event.pos)
            return True
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_from_mouse(event.pos)
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True
        return False

    def set_from_mouse(self, pos):
        x = pos[0]
        t = (x - (self.rect.left + self.knob_radius)) / (self.rect.width - 2*self.knob_radius)
        t = max(0.0, min(1.0, t))
        self.value = self.min + t * (self.max - self.min)


class Checkbox:
    def __init__(self, rect, label, font, checked=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.checked = checked
        # small square inside rect for the box
        self.box_rect = pygame.Rect(self.rect.left + 6, self.rect.top + 6, 20, 20)

    def draw(self, surf):
        # label
        lbl = self.font.render(self.label, True, BLACK)
        surf.blit(lbl, (self.rect.left + 36, self.rect.top + 2))
        # box background
        pygame.draw.rect(surf, CHECK_BG, self.box_rect)
        pygame.draw.rect(surf, CHECK_BORDER, self.box_rect, 2)
        if self.checked:
            # draw check mark (simple)
            cx = self.box_rect.centerx
            cy = self.box_rect.centery
            pygame.draw.line(surf, BLACK, (self.box_rect.left+4, cy), (cx, self.box_rect.bottom-4), 3)
            pygame.draw.line(surf, BLACK, (cx, self.box_rect.bottom-4), (self.box_rect.right-4, self.box_rect.top+4), 3)

    def toggle(self):
        self.checked = not self.checked

    def clicked(self, pos):
        return self.box_rect.collidepoint(pos)


# ---------------------------
# Core stochastic trajectory engine (supports PBC)
# ---------------------------
class SpinChainTrajectory:
    def __init__(self, L, gamma_plus=1.0, gamma_minus=1.0, periodic=False):
        self.L = int(L)
        self.gamma_plus = float(gamma_plus)
        self.gamma_minus = float(gamma_minus)
        self.periodic = bool(periodic)
        self.state = [0] * self.L
        self.sim_time = 0.0
        self.rand = random.Random()
        self.rand.seed()

    def set_initial(self, pattern):
        if pattern == 'all0':
            self.state = [0]*self.L
        elif pattern == 'all1':
            self.state = [1]*self.L
        elif pattern == 'neel':
            self.state = [(i % 2) for i in range(self.L)]
        else:
            self.state = [0]*self.L
        self.sim_time = 0.0

    def set_rates(self, gamma_plus, gamma_minus):
        self.gamma_plus = float(gamma_plus)
        self.gamma_minus = float(gamma_minus)

    def set_periodic(self, periodic):
        self.periodic = bool(periodic)

    def flip_manual(self, pos):
        if 0 <= pos < self.L:
            self.state[pos] ^= 1

    def list_allowed_events(self):
        """Return list of (pos, rate) for allowed center positions where chi_i == 1.
        For open boundaries: i in 0..L-3 and center pos = i+1
        For periodic: i in 0..L-1, center pos = (i+1) % L, neighbors are i and i+2 (mod L)
        """
        events = []
        if self.L < 3:
            return events
        s = self.state
        if not self.periodic:
            # open boundaries: i = 0 .. L-3, center pos = i+1 (1..L-2)
            for i in range(0, self.L - 2):
                if s[i] == 0 and s[i+2] == 0:
                    pos = i + 1
                    rate = self.gamma_plus if s[pos] == 0 else self.gamma_minus
                    if rate > 0.0:
                        events.append((pos, rate))
        else:
            # periodic: consider all i = 0 .. L-1, center pos = (i+1) % L
            L = self.L
            for i in range(0, L):
                left = s[i]
                right = s[(i+2) % L]
                if left == 0 and right == 0:
                    pos = (i + 1) % L
                    rate = self.gamma_plus if s[pos] == 0 else self.gamma_minus
                    if rate > 0.0:
                        events.append((pos, rate))
        return events

    def step_gillespie(self, max_dt):
        """Advance up to max_dt of simulated time via Gillespie events."""
        t_used = 0.0
        budget = float(max_dt)
        while True:
            events = self.list_allowed_events()
            if not events:
                break
            rates = [r for (_, r) in events]
            R = sum(rates)
            if R <= 0.0:
                break
            u = self.rand.random()
            if u <= 1e-16:
                u = 1e-16
            dt = -math.log(u) / R
            if dt > budget:
                t_used += budget
                self.sim_time += budget
                break
            budget -= dt
            t_used += dt
            self.sim_time += dt
            r_pick = self.rand.random() * R
            cum = 0.0
            chosen_idx = None
            for idx, (_, r) in enumerate(events):
                cum += r
                if r_pick <= cum:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                chosen_idx = len(events) - 1
            pos = events[chosen_idx][0]
            # flip center spin
            self.state[pos] ^= 1
        return t_used


# ---------------------------
# Pygame application
# ---------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Spin chain: real-time trajectory — PBC + Speed fixed")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.bigfont = pygame.font.SysFont("Arial", 20, bold=True)

        self.L = DEFAULT_L
        self.model = SpinChainTrajectory(self.L, gamma_plus=1.0, gamma_minus=1.0, periodic=False)
        self.model.set_initial('neel')
        self.running = False

        # Buttons
        self.btn_start = Button((980, 20, 200, 36), "Start", self.font)
        self.btn_pause = Button((980, 64, 200, 36), "Pause", self.font, bg=(240,200,200))
        self.btn_reset = Button((980, 108, 200, 36), "Reset (All 0)", self.font, bg=(200,200,240))
        self.btn_all0 = Button((40, 660, 110, 30), "All 0", self.font)
        self.btn_neel = Button((160, 660, 140, 30), "Neel 0101...", self.font)
        self.btn_all1 = Button((310, 660, 110, 30), "All 1", self.font)

        # Sliders (placed with good spacing to avoid overlap)
        self.slider_gamma_plus = Slider((980, 160, 200, 18), 0.0, 10.0, 1.0, self.font, label='gamma_plus')
        self.slider_gamma_minus = Slider((980, 200, 200, 18), 0.0, 10.0, 1.0, self.font, label='gamma_minus')
        self.slider_L = Slider((980, 240, 200, 18), float(MIN_L), float(MAX_L), float(self.L), self.font, label='L (length)')
        self.slider_speed = Slider((980, 320, 200, 18), 0.0, 200.0, 1.0, self.font, label='Speed')

        # Checkbox for boundary conditions
        self.checkbox_pbc = Checkbox((980, 280, 220, 36), "Periodic Boundary Conditions", self.font, checked=False)

        # drawing area
        self.spin_area = pygame.Rect(30, 20, 920, 620)
        self.update_layout()

    def update_layout(self):
        padding = 2
        if self.L <= 0:
            self.cell_w = 8
        else:
            total_padding = padding * (self.L - 1)
            max_w = self.spin_area.width - total_padding
            size = max(4, max_w // max(1, self.L))
            size = min(size, 40)
            self.cell_w = size
        total_w = self.L * self.cell_w + (self.L - 1) * padding
        start_x = self.spin_area.left + max(0, (self.spin_area.width - total_w) // 2)
        y = self.spin_area.top + (self.spin_area.height - self.cell_w) // 2
        self.positions = [(start_x + i*(self.cell_w + padding), y) for i in range(self.L)]

    def draw_spins(self):
        s = self.model.state
        for i in range(self.L):
            x, y = self.positions[i]
            rect = pygame.Rect(x, y, self.cell_w, self.cell_w)
            color = RED if s[i] == 1 else BLUE
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
        # optional indices
        if self.cell_w >= 10 and self.L <= 60:
            for i in range(self.L):
                x, y = self.positions[i]
                txt = self.font.render(str(i), True, BLACK)
                self.screen.blit(txt, (x, y - 18))

    def compute_allowed_count(self):
        return len(self.model.list_allowed_events())

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.running = not self.running
                elif event.key == pygame.K_r:
                    self.reset_to_preset('all0')

            # slider interactions
            if self.slider_gamma_plus.handle_event(event):
                self.model.set_rates(self.slider_gamma_plus.value, self.slider_gamma_minus.value)
            if self.slider_gamma_minus.handle_event(event):
                self.model.set_rates(self.slider_gamma_plus.value, self.slider_gamma_minus.value)
            if self.slider_L.handle_event(event):
                newL = int(round(self.slider_L.value))
                if newL != self.L:
                    self.L = max(MIN_L, min(MAX_L, newL))
                    # resize model.state preserving or truncating
                    cur = self.model.state
                    if len(cur) < self.L:
                        cur = cur + [0] * (self.L - len(cur))
                    else:
                        cur = cur[:self.L]
                    self.model.L = self.L
                    self.model.state = cur
                    self.update_layout()
            if self.slider_speed.handle_event(event):
                pass

            # checkbox click handling
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self.checkbox_pbc.clicked(pos):
                    self.checkbox_pbc.toggle()
                    self.model.set_periodic(self.checkbox_pbc.checked)

                if self.btn_start.clicked(pos):
                    self.running = True
                elif self.btn_pause.clicked(pos):
                    self.running = False
                elif self.btn_reset.clicked(pos):
                    self.reset_to_preset('all0')
                elif self.btn_all0.clicked(pos):
                    self.reset_to_preset('all0')
                elif self.btn_neel.clicked(pos):
                    self.reset_to_preset('neel')
                elif self.btn_all1.clicked(pos):
                    self.reset_to_preset('all1')
                else:
                    # manual flip while paused
                    if not self.running:
                        for i, (x, y) in enumerate(self.positions):
                            rect = pygame.Rect(x, y, self.cell_w, self.cell_w)
                            if rect.collidepoint(pos):
                                self.model.flip_manual(i)
                                break

    def reset_to_preset(self, name):
        self.model.set_initial(name)
        if self.model.L != self.L:
            self.model.L = self.L
            self.model.state = self.model.state[:self.L] + [0] * max(0, self.L - len(self.model.state))
        self.update_layout()

    def draw_ui(self):
        # control panel background
        pygame.draw.rect(self.screen, (245,245,245), (960, 10, 230, 700), border_radius=8)

        # buttons
        self.btn_start.draw(self.screen)
        self.btn_pause.draw(self.screen)
        self.btn_reset.draw(self.screen)
        self.btn_all0.draw(self.screen)
        self.btn_neel.draw(self.screen)
        self.btn_all1.draw(self.screen)

        # sliders and checkbox
        self.slider_gamma_plus.draw(self.screen)
        self.slider_gamma_minus.draw(self.screen)
        self.slider_L.draw(self.screen)
        # checkbox sits between L and Speed
        self.checkbox_pbc.draw(self.screen)
        self.slider_speed.draw(self.screen)

        # info
        x = 970
        y = 360
        lines = [
            f"L: {self.L}",
            f"Sim time: {self.model.sim_time:.6f}",
            f"gamma_plus: {self.model.gamma_plus:.6g}",
            f"gamma_minus: {self.model.gamma_minus:.6g}",
            f"Running: {'Yes' if self.running else 'No'}",
            f"Allowed events: {self.compute_allowed_count()}",
            f"Boundary: {'Periodic' if self.checkbox_pbc.checked else 'Open'}",
            f"Speed: {self.slider_speed.value:.3g}x",
        ]
        for ln in lines:
            txt = self.font.render(ln, True, BLACK)
            self.screen.blit(txt, (x, y))
            y += 22

        # help text
        help_y = y + 8
        help_lines = [
            "Controls:",
            "- Start / Pause / Reset",
            "- Click square to flip while paused",
            "- Space toggles run; R resets (All 0)",
            "- Toggle 'Periodic Boundary Conditions' box to switch",
            "- Speed multiplies simulated time per real second",
        ]
        for ln in help_lines:
            txt = self.font.render(ln, True, DARK_GRAY)
            self.screen.blit(txt, (x, help_y))
            help_y += 18

    def run(self):
        self.update_layout()
        last = time.time()
        while True:
            self.handle_events()
            now = time.time()
            elapsed = now - last
            last = now

            # update rates in case sliders changed
            self.model.set_rates(self.slider_gamma_plus.value, self.slider_gamma_minus.value)

            if self.running:
                real_elapsed = min(elapsed, 0.2)
                speed = float(self.slider_speed.value)
                sim_budget = real_elapsed * speed
                if sim_budget > 0.0:
                    self.model.step_gillespie(sim_budget)

            # draw frame
            self.screen.fill(WHITE)
            pygame.draw.rect(self.screen, (250,250,250), self.spin_area, border_radius=6)
            pygame.draw.rect(self.screen, DARK_GRAY, self.spin_area, 2, border_radius=6)

            self.draw_spins()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    app = App()
    try:
        app.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)
