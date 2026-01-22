#!/usr/bin/env python3
# ECG ADS1115 + HP/Notch/LP


import os
import time
import signal
import threading
import math
from collections import deque
from ctypes import byref, c_int

import sdl2
import sdl2.sdlttf as sdlttf

try:
    from smbus2 import SMBus
except Exception:
    from smbus import SMBus

os.environ.setdefault("SDL_VIDEODRIVER", "KMSDRM")

# =========================
# Zeitparameter / Anzeige
# =========================
SAMPLE_DT   = 0.005   # 200 Hz target
PLOT_DT     = 0.03    # Aktualisierungsintervall der Anzeige
WINDOW_SEC  = 3.0     # Zeitfenster der dargestellten Kurve
FS = 1.0 / SAMPLE_DT  #Abtastfrequenz

DISPLAY_MA_N = 5
STEP_DRAW = 2
TEXT_ZONE_H = 110

BUF_MAX = 12000
MAX_TAKE_PER_FRAME = 600

# =========================
# ADS1115   Konfiguration
# =========================
I2C_BUS = 1
ADS_ADDR = 0x48

FS_VOLT =4.096   # Vollbereichsspannung
LSB = FS_VOLT / 32768.0  # Auflösung pro Bit

REG_CONV  = 0x00
REG_CFG   = 0x01

# =========================
# 50 Hz Notch
# =========================
NOTCH_F0 = 50.0
NOTCH_Q  = 10.0

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
# Globales Lauf-Flag
# =========================
running = True  #steuert die Hauptschleife des Programms

def on_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, on_sigint)

# ============================================================
# 1) ADS1115 Reader
# ============================================================
class ADS1115Reader:
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
# 2) ECG-Signalverarbeitung
# ============================================================
class ECGProcessor:
    def __init__(self):
        self.reset()
        self.bpm_inst = 0.0

    @staticmethod
    def trimmed_mean(values, trim=1):
        if not values:
            return 0.0
        n = len(values)
        if n <= 2 * trim:
            return sum(values) / n
        v = sorted(values)
        v = v[trim:-trim]
        return sum(v) / len(v)

    def reset(self):
        self.baseline = 0.0
        self.lp = 0.0

        self.SPKI = 0.0
        self.NPKI = 0.0
        self.bpm_inst = 0.0
        self.last_peak_index = -999999
        self.refractory = int(0.45 * FS)

        self.peaks = []
        self.rr_avg = None

        self.dbg_thr1 = 0.0
        self.dbg_mwi  = 0.0
        self.dbg_spki = 0.0
        self.dbg_npki = 0.0
        self.dbg_peaks= 0

        # notch state
        self.n_x1 = self.n_x2 = 0.0
        self.n_y1 = self.n_y2 = 0.0

        # MWI (~80ms)
        self.mwi_win = deque(maxlen=max(1, int(0.08 * FS)))
        self.mwi_sum = 0.0

        # event Erkennung
        self.in_event = False
        self.event_start_i = 0
        self.event_deadline_i = 0
        self.event_max_mwi = 0.0
        self.event_max_absf = 0.0
        self.event_max_absf_i = 0
        self.event_above_count = 0

        self.event_min_len = max(1, int(0.020 * FS))
        self.event_max_len = max(2, int(0.200 * FS))

        # warm-up
        self.warm_n = int(2 * FS)
        self._warm_buf = []
        self._warm_done = False

        # stable display bpm
        self.peak_times = deque()
        self.bpm_window_sec = 8.0
        self.bpm_display = 0.0

        # jump confirmation + asymmetric smoothing
        self.jump_count = 0
        self.jump_confirm_n = 3
        self.jump_threshold_bpm = 15.0
        self.alpha_up = 0.06
        self.alpha_down = 0.25

        # auto recovery
        self.last_accept_i = None
        self.no_peak_recover_sec = 3.0
        self.have_valid_bpm = False

    def _finalize_warmup(self):
        if not self._warm_buf:
            self.NPKI = 0.0
            self.SPKI = 0.0
            self._warm_done = True
            return

        buf = sorted(self._warm_buf)
        n = len(buf)

        def q(p):
            if n == 1:
                return buf[0]
            idx = int(round(p * (n - 1)))
            idx = max(0, min(n - 1, idx))
            return buf[idx]


        self.NPKI = q(0.20)
        self.SPKI = q(0.90)
        if self.SPKI < self.NPKI:
            self.SPKI = self.NPKI
        if self.SPKI == 0.0 and self.NPKI > 0.0:
            self.SPKI = 3.0 * self.NPKI

        self._warm_done = True

    def _do_recover(self):
        self._warm_buf = []
        self._warm_done = False

        self.peaks.clear()
        self.peak_times.clear()
        self.rr_avg = None

        self.bpm_display = 0.0
        self.have_valid_bpm = False
        self.jump_count = 0

        self.last_peak_index = -999999
        self.in_event = False

        self.mwi_win.clear()
        self.mwi_sum = 0.0

        self.last_accept_i = None
        self.bpm_inst = 0.0

    def _update_bpm_display_from_window(self, current_peak_i: int):
        self.peak_times.append(current_peak_i)

        min_i = current_peak_i - int(self.bpm_window_sec * FS)
        while self.peak_times and self.peak_times[0] < min_i:
            self.peak_times.popleft()

        # RR-Erkennung: peak Berechnung
        if len(self.peak_times) < 2:
            return

        rrs = []
        for k in range(1, len(self.peak_times)):
            rr = (self.peak_times[k] - self.peak_times[k - 1]) / FS
            if 0.33 <= rr <= 2.0:
                rrs.append(rr)

        if len(rrs) < 1:
            return

        rr_use = rrs[0] if len(rrs) == 1 else self.trimmed_mean(rrs, trim=1)
        if rr_use <= 0:
            return

        # rr_avg
        if self.rr_avg is not None:
            rrs2 = [rr for rr in rrs if (0.80 * self.rr_avg) <= rr <= (1.25 * self.rr_avg)]
            if rrs2:
                rr_use = rrs2[0] if len(rrs2) == 1 else self.trimmed_mean(rrs2, trim=1)

        bpm_new = 60.0 / rr_use

        # jump confirmation
        if self.bpm_display > 0 and bpm_new > (self.bpm_display + self.jump_threshold_bpm):
            self.jump_count += 1
            if self.jump_count < self.jump_confirm_n:
                return
        else:
            self.jump_count = 0

        # asymmetric smoothing
        if self.bpm_display <= 0:
            self.bpm_display = bpm_new
        else:
            a = self.alpha_up if bpm_new > self.bpm_display else self.alpha_down
            self.bpm_display = a * bpm_new + (1.0 - a) * self.bpm_display

        self.have_valid_bpm = True

    def process(self, voltage: float, i: int, use_highpass: bool, use_notch: bool, use_lowpass: bool):
        if i == 0:
            self.reset()
            self.baseline = voltage

        # High-pass
        if use_highpass:
            hp_win_sec = 1.0
            alpha_hp = SAMPLE_DT / (hp_win_sec + SAMPLE_DT)
            self.baseline += alpha_hp * (voltage - self.baseline)
            hp = voltage - self.baseline
        else:
            hp = voltage

        # Notch
        if use_notch:
            x = hp
            y = _b0 * x + _b1 * self.n_x1 + _b2 * self.n_x2 - _a1 * self.n_y1 - _a2 * self.n_y2
            self.n_x2, self.n_x1 = self.n_x1, x
            self.n_y2, self.n_y1 = self.n_y1, y
            hp = y

        # Low-pass
        if use_lowpass:
            fc = 20.0
            lp_tau = 1.0 / (2.0 * math.pi * fc)
            alpha_lp = SAMPLE_DT / (lp_tau + SAMPLE_DT)
            self.lp += alpha_lp * (hp - self.lp)
            filtered = self.lp
        else:
            filtered = hp

        # Feature: |filtered| + MWI
        amp = abs(filtered)
        if len(self.mwi_win) == self.mwi_win.maxlen:
            self.mwi_sum -= self.mwi_win[0]
        self.mwi_win.append(amp)
        self.mwi_sum += amp
        mwi = self.mwi_sum / len(self.mwi_win)

        # Warm-up
        if not self._warm_done:
            self._warm_buf.append(mwi)
            if len(self._warm_buf) >= self.warm_n:
                self._finalize_warmup()
            return filtered, self.bpm_inst, self.bpm_display

        # Auto recovery:  bpm/peak 
        if self.have_valid_bpm and self.last_accept_i is not None:
            if i - self.last_accept_i > int(self.no_peak_recover_sec * FS):
                self._do_recover()
                return filtered, self.bpm_inst, self.bpm_display

        # Threshold
        THR1 = self.NPKI + 0.12 * (self.SPKI - self.NPKI)

        self.dbg_thr1 = THR1
        self.dbg_mwi  = mwi
        self.dbg_spki = self.SPKI
        self.dbg_npki = self.NPKI
        self.dbg_peaks= len(self.peaks)

        # Event start
        if not self.in_event:
            if (mwi >= THR1) and (i - self.last_peak_index >= self.refractory):
                self.in_event = True
                self.event_start_i = i
                self.event_deadline_i = i + self.event_max_len
                self.event_max_mwi = mwi
                self.event_max_absf = amp
                self.event_max_absf_i = i
                self.event_above_count = 1
            else:
                #  NPKI: event
                self.NPKI = 0.02 * mwi + 0.98 * self.NPKI

            return filtered, self.bpm_inst, self.bpm_display

        # Inside event
        if mwi >= THR1:
            self.event_above_count += 1

        if mwi > self.event_max_mwi:
            self.event_max_mwi = mwi

        if amp > self.event_max_absf:
            self.event_max_absf = amp
            self.event_max_absf_i = i

        end_event = False
        if (mwi < THR1 and (i - self.event_start_i) >= self.event_min_len):
            end_event = True
        if i >= self.event_deadline_i:
            end_event = True

        if not end_event:
            return filtered, self.bpm_inst, self.bpm_display

        # Finalize event
        duration = i - self.event_start_i
        ok_len = (duration <= self.event_max_len)
        ok_above = (self.event_above_count >= self.event_min_len)

        accepted_peak = False
        accepted_peak_i = None

        if ok_len and ok_above:
            peak_i = self.event_max_absf_i

            if peak_i - self.last_peak_index >= self.refractory:
                self.peaks.append(peak_i)
                self.last_peak_index = peak_i

                self.SPKI = 0.125 * self.event_max_mwi + 0.875 * self.SPKI

                if len(self.peaks) >= 2:
                    rr = (self.peaks[-1] - self.peaks[-2]) / FS

                    if not (0.33 <= rr <= 2.0):
                        self.peaks.pop()
                        self.last_peak_index = self.peaks[-1] if self.peaks else -999999
                        self.NPKI = 0.125 * self.event_max_mwi + 0.875 * self.NPKI
                    else:
                        if (self.rr_avg is not None) and (rr < 0.75 * self.rr_avg):
                            self.peaks.pop()
                            self.last_peak_index = self.peaks[-1] if self.peaks else -999999
                            self.NPKI = 0.125 * self.event_max_mwi + 0.875 * self.NPKI
                        else:
                            self.rr_avg = rr if self.rr_avg is None else (0.2 * rr + 0.8 * self.rr_avg)
                            self.bpm_inst = 60.0 / rr
                            accepted_peak = True
                            accepted_peak_i = peak_i
                else:
                    accepted_peak = True
                    accepted_peak_i = peak_i
            else:
                self.NPKI = 0.125 * self.event_max_mwi + 0.875 * self.NPKI
        else:
            self.NPKI = 0.125 * self.event_max_mwi + 0.875 * self.NPKI

        if accepted_peak and accepted_peak_i is not None:
            self.last_accept_i = accepted_peak_i
            self._update_bpm_display_from_window(accepted_peak_i)

        self.in_event = False
        return filtered, self.bpm_inst, self.bpm_display

# ============================================================
# 3) ECG App (SDL UI)
# ============================================================
class ECGApp:
    def __init__(self):
        self.paused = False

        self.use_highpass = True
        self.use_notch = True
        self.use_lowpass = True

        self.buf = deque()
        self.eff_rate_hz = 0.0

        self.ads = None
        self.proc = ECGProcessor()

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

        self.SCALE = 25

        self.text_items = []
        self.last_text_t = 0.0
        self.last_idx = 0
        self.log_signal= []

        self.sampler_thread = None

        # Restart flag:
        self.restart_evt = threading.Event()

    @staticmethod
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
        GRID_step = 40
        mid = self.plot_h // 2
        base = int(round(mid / GRID_step) * GRID_step)
        y = base - int(v * self.SCALE)
        if y < 0:
            return 0
        if y >= self.plot_h:
            return self.plot_h - 1

        return y

    def _save_signal_file(self):
         if not self.log_signal:
              return

         subject="test1"
         ts= time.strftime("%Y%m%d_%H%M%S")
         fname = f"ekg_{subject}_{ts}.csv"

         with open(fname, "w") as f:
              f.write("time_s,ekg_V\n")
              for t, V in self.log_signal:
                   f.write(f"{t:.6f},{V:.6f}\n")

         print(f"EKG signal hat gespeichert in {fname}")



    def _sampler_worker(self):
        global running

        i = 0
        next_t = time.perf_counter()

        last_rate_t = time.perf_counter()
        count = 0

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

            # Restart in place
            if self.restart_evt.is_set():
                self.restart_evt.clear()
                try:
                    self.buf.clear()
                except Exception:
                    pass
                i = 0
                count = 0
                last_rate_t = perf()
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

            slp = next_t - perf()
            if slp > 0:
                sleepf(slp)
            else:
                next_t = perf()

    def _init_hw(self):
        self.ads = ADS1115Reader(I2C_BUS, ADS_ADDR)
        _ = self.ads.read_voltage()
        print("ADS1115 verbunden")

    def _init_sdl(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise RuntimeError("SDL_Init failed: " + sdl2.SDL_GetError().decode())
        sdlttf.TTF_Init()

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

    def _restart_everything_in_place(self):

        self.restart_evt.set() 
        self.buf.clear()

        self.proc.reset()

        self.ecg_disp.clear()
        self.ma_buf.clear()
        self.ma_sum = 0.0
        self.last_idx = 0

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

                if key == sdl2.SDLK_h:
                    self.use_highpass = not self.use_highpass
                if key == sdl2.SDLK_n:
                    self.use_notch = not self.use_notch
                if key == sdl2.SDLK_l:
                    self.use_lowpass = not self.use_lowpass

                #  Restart-in-place (R)
                if key == sdl2.SDLK_r:
                    self._restart_everything_in_place()

    def _update_text(self, now):
        if now - self.last_text_t < 0.5:
            return
        self.last_text_t = now

        self._destroy_text_items(self.text_items)
        self.text_items = []

        t_show = self.last_idx * SAMPLE_DT

        #ا bpm_displa : bpm_inst
        if self.proc.bpm_display > 0:
            bpm_show = int(round(self.proc.bpm_display))
        elif self.proc.bpm_inst > 0:
            bpm_show = int(round(self.proc.bpm_inst))
        else:
            bpm_show = 0

        x_text = 20
        y0 = self.H - 60
        dy = 18

        line1 = f"t={t_show:6.2f}s   BPM={bpm_show:3d}   peaks={self.proc.dbg_peaks}   EffRate={self.eff_rate_hz:5.1f} Hz"
        t1, r1 = self._make_text_texture(self.ren, self.font, line1, x_text, y0, (255, 255, 255))

        line2 = (
            f"HP(H):{'Ein' if self.use_highpass else 'Aus'}   "
            f"Notch(N):{'Ein' if self.use_notch else 'Aus'}   "
            f"LP(L):{'Ein' if self.use_lowpass else 'Aus'}   Restart(R)"
        )
        t2, r2 = self._make_text_texture(self.ren, self.font, line2, x_text, y0 + dy, (180, 180, 180))

        self.text_items = [(t1, r1), (t2, r2)]

     #   line3 = (
      #      f"warm={int(self.proc._warm_done)} wlen={len(self.proc._warm_buf)} "
       #     f"mwi={self.proc.dbg_mwi:.6f} thr1={self.proc.dbg_thr1:.6f} peaks={self.proc.dbg_peaks}"
       # )
        #t3, r3 = self._make_text_texture(self.ren, self.font, line3, x_text, y0 + 2*dy, (140, 140, 255))
       # self.text_items.append((t3, r3))

        if self.paused:
            tp, rp = self._make_text_texture(self.ren, self.font, "PAUSE (P)", x_text, y0 - dy, (255, 100, 100))
            self.text_items.append((tp, rp))

    def _draw_frame(self, now):
        if self.grid_tex:
            sdl2.SDL_RenderCopy(self.ren, self.grid_tex, None, None)
        else:
            sdl2.SDL_SetRenderDrawColor(self.ren, 0, 0, 40, 255)
            sdl2.SDL_RenderClear(self.ren)

        sdl2.SDL_SetRenderDrawColor(self.ren, 0, 255, 0, 255)
        n = len(self.ecg_disp)
        step = max(1, STEP_DRAW)

        if n >= 2:
            x_start = self.PW - n
            if x_start < 0:
                x_start = 0

            for i in range(0, n - 1, step):
                j = i + step
                if j >= n:
                    j = n - 1

                x1 = self.xs[x_start + i]
                x2 = self.xs[x_start + j]
                y1 = self._toY(self.ecg_disp[i] * 40)
                y2 = self._toY(self.ecg_disp[j] * 40)
                sdl2.SDL_RenderDrawLine(self.ren, x1, y1, x2, y2)

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
            print("ADS1115 SMBus funktioniert nicht:", repr(e))
            return

        try:
            self._init_sdl()
        except Exception as e:
            print("SDL init nicht funktioniert:", repr(e))
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

                        filt, _binst, _bdisp = self.proc.process(
                            v, idx,
                            self.use_highpass,
                            self.use_notch,
                            self.use_lowpass
                        )

                        t_sec= idx * SAMPLE_DT
                        self.log_signal.append((t_sec, filt))

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

        self._save_signal_file()

        print("Programm beendet.")

if __name__ == "__main__":
    ECGApp().run()
