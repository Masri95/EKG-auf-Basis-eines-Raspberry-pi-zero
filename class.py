import os
import time
import signal
import threading
import math
from collections import deque
from ctypes import byref, c_int

import sdl2
import sdl2.sdlttf as sdlttf

# -------------------------
# SMBus import (smbus2)
# -------------------------
try:
    from smbus2 import SMBus
except Exception:
    from smbus import SMBus

os.environ.setdefault("SDL_VIDEODRIVER", "KMSDRM")

# =========================
# Timing / Anzeige
# =========================
SAMPLE_DT   = 0.005   # 200 Hz target
PLOT_DT     = 0.12  # Darstellung ca. alle 120 ms (~8.3 FPS)
WINDOW_SEC  = 1.0
FS = 1.0 / SAMPLE_DT # Abtastfrequenz (Anzahl der Samples pro Sekunde)

DISPLAY_MA_N = 7  # Moving Average nur für die Anzeige (7 Samples)
STEP_DRAW = 8

TEXT_ZONE_H = 110

BUF_MAX = 12000  ## Max. Anzahl gespeicherter Samples im Buffer (deque)
MAX_TAKE_PER_FRAME = 300 ## Max. 300 Samples aus dem Puffer für Verarbeitung/Anzeige

# =========================
# ADS1115 SMBus settings
# =========================
I2C_BUS = 1
ADS_ADDR = 0x48

# PGA = ±4.096V
FS_VOLT = 4.096
LSB = FS_VOLT / 32768.0

# ADS1115 Registers
REG_CONV  = 0x00
REG_CFG   = 0x01

# =========================
# 50 Hz Notch Filter (IIR biquad)
# =========================
NOTCH_F0 = 50.0   # Hz
NOTCH_Q  = 30.0

_w0 = 2.0 * math.pi * (NOTCH_F0 / FS)
_alpha = math.sin(_w0) / (2.0 * NOTCH_Q)

_b0 = 1.0
_b1 = -2.0 * math.cos(_w0)
_b2 = 1.0
_a0 = 1.0 + _alpha
_a1 = -2.0 * math.cos(_w0)
_a2 = 1.0 - _alpha

_b0 /= _a0
_b1 /= _a0
_b2 /= _a0
_a1 /= _a0
_a2 /= _a0


# =========================
# Globales Lauf-Flag (SIGINT)
# =========================
running = True

def on_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, on_sigint)


# ============================================================
# 1) ADS1115 Reader (Hardware)
# ============================================================
class ADS1115Reader:
"""
Konfiguriert den ADS1115 einmal im Continuous-Mode mit 860 SPS.
Anschließend wird nur noch das Conversion-Register ausgelesen (schneller Zugriff).
"""
    def __init__(self, bus_id=1, addr=0x48):
        self.bus = SMBus(bus_id)
        self.addr = addr
        self._configure_continuous()

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    @staticmethod
    def _swap16(x: int) -> int:
        return ((x & 0xFF) << 8) | ((x >> 8) & 0xFF)

    def _write_config(self, cfg: int) -> None:
        self.bus.write_word_data(self.addr, REG_CFG, self._swap16(cfg))

    def _configure_continuous(self) -> None:
        """
        OS=1, MUX=AIN0-GND, PGA=±4.096V, MODE=continuous, DR=860SPS, comparator disabled
        """
        OS   = 1 << 15
        MUX_AIN0 = 0b100 << 12
        PGA_4V096 = 0b001 << 9
        MODE_CONT = 0 << 8
        DR_860 = 0b111 << 5
        COMP_QUE_DISABLE = 0b11

        cfg = OS | MUX_AIN0 | PGA_4V096 | MODE_CONT | DR_860 | COMP_QUE_DISABLE
        self._write_config(cfg)
        time.sleep(0.01)

    def read_voltage(self) -> float:
        w = self.bus.read_word_data(self.addr, REG_CONV)
        w = self._swap16(w)
        if w & 0x8000:
            w -= 0x10000
        return w * LSB


# ============================================================
# 2) ECG Processor (HP + Notch + LP + BPM)
# ============================================================
class ECGProcessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.baseline = 0.0
        self.lp = 0.0

        self.SPKI = 0.0
        self.NPKI = 0.0
        self.last_peak_index = -999999
        self.refractory = int(0.200 * FS) # Multiplikation mit FS zur Umrechnung von 0,200 s in Anzahl der Samples

        self.peaks = []
        self.bpm_inst = 0.0
        self.bpm_smooth = 0.0

        # notch biquad state
        self.n_x1 = 0.0
        self.n_x2 = 0.0
        self.n_y1 = 0.0
        self.n_y2 = 0.0

    def process(self, voltage: float, i: int, use_highpass: bool, use_notch: bool, use_lowpass: bool):
        if i == 0:
            self.reset()
            self.baseline = voltage

        # High-pass
        if use_highpass:
            hp_win_sec = 0.3
            alpha_hp = SAMPLE_DT / hp_win_sec
            self.baseline += alpha_hp * (voltage - self.baseline)
            hp = voltage - self.baseline
        else:
            hp = voltage

        # 50 Hz Notch
        if use_notch:
            x = hp
            y = _b0*x + _b1*self.n_x1 + _b2*self.n_x2 - _a1*self.n_y1 - _a2*self.n_y2
            self.n_x2, self.n_x1 = self.n_x1, x
            self.n_y2, self.n_y1 = self.n_y1, y
            hp = y

        # Low-pass 
        if use_lowpass:
            lp_tau = 0.04
            alpha_lp = SAMPLE_DT / lp_tau
            self.lp += alpha_lp * (hp - self.lp)
            filtered = self.lp
        else:
            filtered = hp

        # Peak/BPM logik
        energy = abs(filtered)

        if i < int(2 * FS):
            self.SPKI = max(self.SPKI, energy)   #Signal Peak Nivau
            self.NPKI = (self.NPKI * i + energy) / (i + 1) #Noise Peak Nivau
            return filtered, self.bpm_inst, self.bpm_smooth

        THR1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)  #adaptive thresholding

        if (energy >= THR1) and (i - self.last_peak_index >= self.refractory):
            self.last_peak_index = i
            self.peaks.append(i)
            self.SPKI = 0.125 * energy + 0.875 * self.SPKI   #Exponential Moving Average (EMA)

            if len(self.peaks) >= 2:
                rr = (self.peaks[-1] - self.peaks[-2]) / FS
                if rr > 0:
                    self.bpm_inst = 60.0 / rr
                    self.bpm_smooth = (0.3 * self.bpm_inst + 0.7 * self.bpm_smooth) if self.bpm_smooth > 0 else self.bpm_inst
        else:
            self.NPKI = 0.125 * energy + 0.875 * self.NPKI

        return filtered, self.bpm_inst, self.bpm_smooth


# ============================================================
# 3) ECG App (Thread + Buffer + SDL UI)
# ============================================================
class ECGApp:
    def __init__(self):
        self.paused = False

        # Filter-Schalter
        self.use_highpass = True
        self.use_notch = True
        self.use_lowpass = True

        self.buf = deque()
        self.eff_rate_hz = 0.0 #Effective sampling

        self.ads = None #lazy initialization.
        self.proc = ECGProcessor()

        # SDL stuff
        self.window = None
        self.ren = None
        self.font = None
        self.grid_tex = None

        self.W = 0
        self.H = 0
        self.plot_h = 0

        self.PW = max(2, int(WINDOW_SEC * FS))
        self.xs = None

        self.ecg_disp = deque(maxlen=self.PW)
        self.ma_buf = deque(maxlen=DISPLAY_MA_N)
        self.ma_sum = 0.0

        self.SCALE = 40 

        self.text_items = []
        self.last_text_t = 0.0
        self.last_idx = 0

        self.sampler_thread = None

    # -------------------------
    # SDL-Helfer
    # -------------------------
    @staticmethod      # unabhängig von der Klasseninstanz
   # Vorab-Rendering statischer Elemente
    def _create_grid_texture(ren, W, H, step=40):
        grid_tex = sdl2.SDL_CreateTexture(
            ren, sdl2.SDL_PIXELFORMAT_RGBA8888, sdl2.SDL_TEXTUREACCESS_TARGET, W, H
        )
        if not grid_tex:
            return None

        sdl2.SDL_SetRenderTarget(ren, grid_tex)
        sdl2.SDL_SetRenderDrawColor(ren, 0, 0, 40, 255)
        sdl2.SDL_RenderClear(ren)

        sdl2.SDL_SetRenderDrawColor(ren, 30, 30, 80, 255)
        for x in range(0, W, step):
            sdl2.SDL_RenderDrawLine(ren, x, 0, x, H - 1)
        for y in range(0, H, step):
            sdl2.SDL_RenderDrawLine(ren, 0, y, W - 1, y)

        sdl2.SDL_SetRenderTarget(ren, None)
        return grid_tex

    @staticmethod
    def _make_text_texture(ren, font, text, x, y, color=(255, 255, 255)):
        if not font:
            return None, None
        r, g, b = color
        surf = sdlttf.TTF_RenderUTF8_Blended(font, text.encode("utf-8"), sdl2.SDL_Color(r, g, b))
        if not surf:
            return None, None
        tex = sdl2.SDL_CreateTextureFromSurface(ren, surf)
        rect = sdl2.SDL_Rect(x, y, surf.contents.w, surf.contents.h)
        sdl2.SDL_FreeSurface(surf)
        return tex, rect

    @staticmethod
    def _destroy_text_items(items):
        for tex, _rect in items:
            if tex:
                sdl2.SDL_DestroyTexture(tex)

    def _toY(self, v: float) -> int:
        mid = self.plot_h // 2
        y = mid - int(v * self.SCALE)
        if y < 0:
            return 0
        if y >= self.plot_h:
            return self.plot_h - 1
        return y

    # -------------------------
    # Sampling thread
    # -------------------------
    def _sampler_worker(self):
        global running

        i = 0
        next_t = time.perf_counter()

        last_rate_t = time.perf_counter()
        count = 0
        
        # Optimierung durch lokale Variablenbindung
        buf_append = self.buf.append
        buf_popleft = self.buf.popleft
        perf = time.perf_counter
        sleepf = time.sleep

        read_v = self.ads.read_voltage

        while running:
            if self.paused:
                sleepf(0.01)
                next_t = perf()
                continue

            next_t += SAMPLE_DT

            try:
                v = read_v()
            except Exception:
                v = 0.0

            buf_append((i, v))
            if len(self.buf) > BUF_MAX:
                buf_popleft()

            i += 1
            count += 1

            now = perf()
            dt = now - last_rate_t
            if dt >= 2.0:
                self.eff_rate_hz = count / dt if dt > 0 else 0.0
                count = 0
                last_rate_t = now
                
          #resynchronization
            slp = next_t - perf()
            if slp > 0:
                sleepf(slp)
            else:
                next_t = perf()

    # -------------------------
    # Init / Run / Cleanup
    # -------------------------
    def _init_hw(self):
        self.ads = ADS1115Reader(I2C_BUS, ADS_ADDR)
        _ = self.ads.read_voltage()
        print(" ADS1115 (SMBus) verbunden.")
        
# Fehler bei der Initialisierung von SDL
    def _init_sdl(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError("SDL_Init failed: " + sdl2.SDL_GetError().decode())
        sdlttf.TTF_Init()
        
# Fehler beim Erstellen des Fensters
        self.window = sdl2.SDL_CreateWindow(
            b"ECG ADS1115 (SMBus + HP/Notch/LP)",
            sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED,
            0, 0, sdl2.SDL_WINDOW_FULLSCREEN
        )
        if not self.window:
            raise RuntimeError("CreateWindow failed: " + sdl2.SDL_GetError().decode())

        Wc, Hc = c_int(), c_int()
        sdl2.SDL_GetWindowSize(self.window, byref(Wc), byref(Hc))
        self.W, self.H = Wc.value, Hc.value
        self.plot_h = max(50, self.H - TEXT_ZONE_H)

        self.ren = sdl2.SDL_CreateRenderer(self.window, -1, sdl2.SDL_RENDERER_SOFTWARE)
        if not self.ren:
            self.ren = sdl2.SDL_CreateRenderer(self.window, -1, sdl2.SDL_RENDERER_ACCELERATED)
        if not self.ren:
            raise RuntimeError("CreateRenderer failed: " + sdl2.SDL_GetError().decode())

        font_path = b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.font = None
        if sdl2.SDL_RWFromFile(font_path, b"rb"):
            self.font = sdlttf.TTF_OpenFont(font_path, 14)

        self.grid_tex = self._create_grid_texture(self.ren, self.W, self.H, step=40)
        self.xs = [int(i * (self.W - 1) / (self.PW - 1)) for i in range(self.PW)]

    def _handle_events(self):
        global running
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(byref(event)) != 0:
            if event.type == sdl2.SDL_KEYDOWN:
                key = event.key.keysym.sym

                if key in (sdl2.SDLK_ESCAPE, sdl2.SDLK_q):
                    running = False
                    break

                if key == sdl2.SDLK_p:
                    self.paused = not self.paused

               # Filter-Schalter
                if key == sdl2.SDLK_h:
                    self.use_highpass = not self.use_highpass
                if key == sdl2.SDLK_n:
                    self.use_notch = not self.use_notch
                if key == sdl2.SDLK_l:
                    self.use_lowpass = not self.use_lowpass

    def _update_text(self, now):
       # Aktualisierung alle 500 ms (TTF-Font-Rendering teuer)
        if now - self.last_text_t < 0.5:
            return
        self.last_text_t = now

        self._destroy_text_items(self.text_items)
        self.text_items = []

        t_show = self.last_idx * SAMPLE_DT
        bpm_show = self.proc.bpm_smooth if self.proc.bpm_smooth > 0 else self.proc.bpm_inst
        if bpm_show <= 0:
            bpm_show = 0.0

        x_text = 20
        y0 = self.H - 90
        dy = 18

        # Line 1: time + bpm + eff rate 
        line1 = f"t={t_show:6.2f}s   BPM={bpm_show:5.1f}   EffRate={self.eff_rate_hz:5.1f} Hz"
        t1, r1 = self._make_text_texture(self.ren, self.font, line1, x_text, y0, (255, 255, 255))

        # Line 2: Filter states
        line2 = (
            f"HP(H):{'ON' if self.use_highpass else 'OFF'}   "
            f"Notch(N):{'ON' if self.use_notch else 'OFF'}   "
            f"LP(L):{'ON' if self.use_lowpass else 'OFF'}    "
        )
        t2, r2 = self._make_text_texture(self.ren, self.font, line2, x_text, y0 + dy, (180, 180, 180))

        self.text_items = [(t1, r1), (t2, r2)]

        if self.paused:
            tp, rp = self._make_text_texture(self.ren, self.font, "PAUSE (P)", x_text, y0 - dy, (255, 100, 100))
            self.text_items.append((tp, rp))

    def _draw_frame(self, now):
        # Hintergrunddarstellung
        if self.grid_tex:
            sdl2.SDL_RenderCopy(self.ren, self.grid_tex, None, None)
        else:
            sdl2.SDL_SetRenderDrawColor(self.ren, 0, 0, 40, 255)
            sdl2.SDL_RenderClear(self.ren)

        # Signalverlauf
        sdl2.SDL_SetRenderDrawColor(self.ren, 0, 255, 0, 255)
        n = len(self.ecg_disp)
        step = max(1, STEP_DRAW)

        if n >= 2:
            x_start = self.PW - n
            if x_start < 0:
                x_start = 0
            last_i = n - 1 - step
            for i in range(0, last_i, step):
                x1 = self.xs[x_start + i]
                x2 = self.xs[x_start + i + step]
                y1 = self._toY(self.ecg_disp[i])
                y2 = self._toY(self.ecg_disp[i + step])
                sdl2.SDL_RenderDrawLine(self.ren, x1, y1, x2, y2)

        # text
        self._update_text(now)
        for tex, rect in self.text_items:
            if tex and rect:
                sdl2.SDL_RenderCopy(self.ren, tex, None, rect)

        sdl2.SDL_RenderPresent(self.ren)

    def run(self):
        global running

        try:
            self._init_hw()
        except Exception as e:
            print("ADS1115 SMBus-Initialisierung/Lesen fehlgeschlagen: ", repr(e))
            print("Hinweis: Ist I2C aktiviert und die Adresse korrekt? ")
            return

        try:
            self._init_sdl()
        except Exception as e:
            print("Fehler bei der Initialisierung von SDL: ", repr(e))
            try:
                if self.ads:
                    self.ads.close()
            except Exception:
                pass
            return

        self.sampler_thread = threading.Thread(target=self._sampler_worker, daemon=True)
        self.sampler_thread.start()

        last_plot_t = time.perf_counter()

        try:
            while running:
                self._handle_events()
                if not running:
                    break

                now = time.perf_counter()
                if now - last_plot_t >= PLOT_DT:
                    last_plot_t += PLOT_DT

                    n_take = min(len(self.buf), MAX_TAKE_PER_FRAME)
                    for _ in range(n_take):
                        idx, v = self.buf.popleft()
                        self.last_idx = idx

                        filt, _bi, _bs = self.proc.process(
                            v, idx,
                            self.use_highpass,
                            self.use_notch,
                            self.use_lowpass
                        )

                       # Glättung der Anzeige mittels gleitendem Mittelwert (Moving Average)
                        if len(self.ma_buf) == self.ma_buf.maxlen:
                            self.ma_sum -= self.ma_buf[0]
                        self.ma_buf.append(filt)
                        self.ma_sum += filt
                        self.ecg_disp.append(self.ma_sum / len(self.ma_buf))

                    self._draw_frame(now)

                time.sleep(0.001)

        finally:
            self.shutdown()

    def shutdown(self):
        global running
        running = False
        time.sleep(0.05)

        self._destroy_text_items(self.text_items)

        if self.grid_tex:
            sdl2.SDL_DestroyTexture(self.grid_tex)
            self.grid_tex = None
        if self.font:
            sdlttf.TTF_CloseFont(self.font)
            self.font = None

        try:
            sdlttf.TTF_Quit()
        except Exception:
            pass

        if self.ren:
            sdl2.SDL_DestroyRenderer(self.ren)
            self.ren = None
        if self.window:
            sdl2.SDL_DestroyWindow(self.window)
            self.window = None
        try:
            sdl2.SDL_Quit()
        except Exception:
            pass

        if self.ads:
            self.ads.close()
            self.ads = None

        print("Programm beendet.")


if __name__ == "__main__":
    ECGApp().run()

